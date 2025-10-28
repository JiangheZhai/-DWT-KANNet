import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import wtconv.util.wavelet as wavelet
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummaryX import summary
from ConvKAN import *
from fast_kan import Fast_KANLinear
from fast_kan import ConvKAN1d  

class groupspectralfusion(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(groupspectralfusion, self).__init__()
        
        self.out_channels = out_channels
        self.bias = bias
        
        self.conv_11=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=out_channels, bias=bias)
        #三种尺寸的空洞卷积
        self.DalConvs_1=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=out_channels, bias=bias)       
        self.DalConvs_2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2, groups=out_channels, bias=bias)      
        self.DalConvs_3=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, stride=1, dilation=3, groups=out_channels, bias=bias)      

  
        self.ln = nn.LayerNorm([out_channels, 12, 12])  
        self.act = nn.GELU()
       
        
    def forward(self, x):                      
        x=self.act(self.ln(self.conv_11(x)))
        #空洞卷积    GELU
        
        x_1=self.DalConvs_1(x)
        x_2=self.DalConvs_2(x + x_1)
        x_3=self.DalConvs_3(x + x_2)
        
        x_conv=self.act(self.ln(x_1+x_2+x_3))        
        return x_conv                            #shape=(b, group, h, w)将维度从c降低到group

#使用简单1×1卷积进行消融对比
# class groupspectralfusion(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(groupspectralfusion, self).__init__()
        
#         self.out_channels = out_channels
#         self.bias = bias
        
#         self.conv_11=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=out_channels, bias=bias)
#         #三种尺寸的空洞卷积
       
#         self.ln = nn.LayerNorm([out_channels, 12, 12])  
#         self.act = nn.GELU()
       
        
#     def forward(self, x):                      
#         x=self.act(self.ln(self.conv_11(x)))
#         #空洞卷积    
   
#         return x   
    
#分支1：小波卷积提取空间特征    
class wavelettransformer(nn.Module):
    def __init__(self, out_channels, wt_levels=3, wt_type='haar'):
        super(wavelettransformer, self).__init__()
        self.wt_levels = wt_levels
        self.out_channels = out_channels
        
        #小波滤波器/逆滤波器
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, out_channels, out_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
 
        self.fourheftconv = nn.ModuleList([fourheftconv(out_channels,3),fourheftconv(out_channels,3),fourheftconv(out_channels,1)])
        #原始输入和小波卷积结果拼接
        
        self.base_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=out_channels, bias=True)
        self.convbase = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=out_channels, bias=True) 

        self.ln = nn.LayerNorm([out_channels, 12, 12])  
        self.act = nn.GELU()
    def forward(self, x):            #x的输入形状为（b, group, h, w）   group是降维后的光谱数
        
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        
        curr_x_ll = x    #初始化当前的低频分量为输入数据
        # 多级小波分解过程
        for i in range(self.wt_levels):
            # 记录当前低频分量的形状
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # 如果当前低频分量的高度或宽度为奇数，则进行零填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)    
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            # 将当前低频分量进行小波变换并提取低频分量
            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)    
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape                            
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  #将高频分量展平
            fourheftconv_i = self.fourheftconv[i]
            curr_x_tag =fourheftconv_i(curr_x_tag)                            
            
            #处理后的结果
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])     #对应YLL
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])    #对应YH

        next_x_ll = 0
        
        for i in range(self.wt_levels-1, -1, -1):   
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)           #shape=(b, group, 4, h/2, w/2)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)  #shape=(b, group, h, w)       
 
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            next_x_ll= self.base_conv(next_x_ll) 
      

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        x = self.act(self.ln(self.convbase(x)))   
        x_spatial = x + x_tag
        
        return x_spatial                     #shape=(b, group, h, w)

class fourheftconv(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(fourheftconv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        #3×3分组卷积层，每个通道都进行单独卷积，因此不涉及到光谱信息的提取
        self.convs =nn.Conv2d(out_channels*4, out_channels*4, kernel_size=kernel_size, padding='same', stride=1, groups=out_channels*4, bias=True)                                                 

        self.BN = nn.BatchNorm2d(out_channels*4)   
        #1×1分组卷积层
        
        #版本2
        self.conv_11 =  nn.ModuleList(
             [nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=out_channels, bias=True) for _ in range(6)]     
        )
        #6→4 1×1卷积层
        self.conv_6dim4 = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=(0, 1, 1), stride=1, bias=True)
        self.BN3d = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self,x):   #x的输入形状为（b, group*4, h/2, w/2）
        b,c,h,w = x.size()
        x_33_conv = self.act(self.BN(self.convs(x)))
        x_33_conv = x_33_conv.reshape(b, self.out_channels, -1, h, w)       #shape=(b, group, 4, h/2, w/2)
        
        #密集链接块的消融实验
        #将4个分量两两相加在进行1×1卷积
        x_11_conv = []
      
       
        #版本2
        for i in range(3):
            for j in range(i+1,4):
                x_one = x_33_conv[:,:,i,:,:]
                x_two = x_33_conv[:,:,j,:,:]
                x_11 =  x_one + x_two   
                conv_ij_11 = self.conv_11[(i+j-1) if i == 0 else (i+j)]
                x_11 = conv_ij_11(x_11)   
                x_11_conv.append(x_11)    
        
        x_11_conv = self.act(self.BN3d(torch.stack(x_11_conv, dim=2)))  #shape = (B, groups, 6, h, w)
        x_11_conv = (self.conv_6dim4(x_11_conv))       
        
        return x_11_conv   #shape=(b, group, 4, h/2, w/2)
        
#小波卷积后的放缩层   替换成Batchnorm
class _ScaleModule(nn.Module):                          
    def __init__(self, dim, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dim
        self.weight = nn.Parameter(torch.ones(*dim) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


# 分支2：KAN结合组注意力机制提取光谱特征
class GSC_kan(nn.Module):
    def __init__(self, dim_in, dim_out, num_groups):          #dim_in = dim_out = group = 32
        super().__init__()
        self.gpwc_kan = groupConvKAN(dim_in, dim_out, num_groups, kernel_size=1)
        self.gc_kan = groupConvKAN(dim_out, dim_out, num_groups, kernel_size=3, padding=1)
        #self.gpwc = nn.Conv2d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        #self.gc = nn.Conv2d(dim_out, dim_out, kernel_size=3, groups=num_groups, padding=1, stride=1)
        
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):                                           #输入x的形状为（b, group, h, w）       group是降维后的光谱数，暂定为32
        
        x=self.gpwc_kan(x)
        x=self.gc_kan(x)
        #x=self.gpwc(x)   
        #x=self.gc(x)  
        
        return self.bn(x)  

class GSSA_kan(nn.Module):
    def __init__(
            self,
            dim,
            patch_size,
            heads,
            dim_head,
            dropout=0.,
            group_spatial_size=4                             #暂定的每个patch：h=w=12，那么group_spatial_size=4，一共会分成9组
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.group_spatial_size = group_spatial_size
        inner_dim = dim_head * heads                         #inner_dim=groups
        self.dim = dim
        n=(patch_size//group_spatial_size)**2
        
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        #self.to_qkv = nn.Conv1d(dim, inner_dim * 3,1,bias=False)    #传统
        self.to_qkv = ConvKAN1d(group_spatial_size*group_spatial_size+1 , 3)   #kan  
        self.group_tokens = nn.Parameter(torch.randn(dim))

        self.group_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),           #对输入数据最后一个维度（大小等于dim_head）进行归一化
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),   #shape (b, heads, num_groups, dim_head) → (b, heads*dim_head, num_groups)   #本来每个通道（num_groups个组）内的group_token都是一个值，但下面经过全连接层后，每个通道内的group_token就不一样了。
            #nn.Conv1d(inner_dim, inner_dim * 2,1),   #传统
            ConvKAN1d( n , 2),   #kans
            Rearrange('b (h c) n -> b h n c', h=heads),
        )
        self.group_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            ConvKAN(inner_dim, dim, 1),
            #nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):               #输入x的形状为（b, groups, h, w）       

        batch, height, width, heads, gss = x.shape[0], *x.shape[-2:], self.heads, self.group_spatial_size    #32个通道，分成8个头，每个头4个通道。
        assert (height % gss) == 0 and (width % gss) == 0, f'height {height} and width {width} must be divisible by group spatial size {gss}'
       

        x = rearrange(x, 'b c (h g1) (w g2) -> (b h w) c (g1 g2)', g1=gss, g2=gss)           #窗口化的通道注意力机制
        w = repeat(self.group_tokens, 'c -> b c 1', b=x.shape[0])                            #因为相同通道的特征图共享一个group_token，所以就相当于全局的通道权重
        x = torch.cat((w, x), dim=-1)         #shape (b h w), dim, 1+g1*g2
        
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))   #shape (b h w), heads, 1+g1*g2, dim_head
        q = q * self.scale
        q = q.transpose(-2, -1)   #q: shape (b h w), heads, dim_head, 1+g1*g2
        dots = einsum('b h i n , b h n j -> b h i j', q, k)        #shape (b h w), heads, dim_head, dim_head 
        attn = self.attend(dots)       
        v = v.transpose(-2, -1 )   #v: shape (b h w), heads, dim_head, 1+g1*g2
        out = torch.matmul(attn, v)  #shape (b h w), heads, dim_head, 1+g1*g2           逻辑通顺，得到通道子注意力机制的输出
        out = out.transpose(-2, -1)   #shape (b h w), heads, 1+g1*g2, dim_head                   


        group_tokens, grouped_fmaps = out[:, :, 0], out[:, :, 1:]
        group_tokens = rearrange(group_tokens, '(b x y) h d -> b h (x y) d', x=height // gss, y=width // gss)        #shape (b, heads, num_groups, dim_head)
        grouped_fmaps = rearrange(grouped_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // gss, y=width // gss)  #shape (b, heads, num_groups, g1*g2, dim_head)
        w_q, w_k = self.group_tokens_to_qk(group_tokens).chunk(2, dim=-1)    #shape (b, heads, num_groups, dim_head) 
        w_q = w_q * self.scale
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)  #shape (b, heads, num_groups, num_groups)
        w_attn = self.group_attend(w_dots)
        aggregated_grouped_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, grouped_fmaps)  #shape (b, heads, num_groups, g1*g2, dim_head)
        fmap = rearrange(aggregated_grouped_fmap, 'b h (x y) (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                         y=width // gss, g1=gss, g2=gss)    #shape=(b, inner_dim, h, w)
        
        
        return self.to_out(fmap)    #shape=(b, group, h, w)
#自定义的通道正则化
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):          
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn    
    def forward(self, x):
        return self.fn(self.norm(x)) 
    
class Transformer_kan(nn.Module):    
    def __init__(
            self,
            dim,
            patch_size,
            dim_head,
            heads,
            dropout=0.1,
            norm_output=True,
            groupsize=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([]) 
        self.layers.append(
                PreNorm(dim, GSSA_kan(dim,patch_size=patch_size, group_spatial_size=groupsize, heads=heads, dim_head=dim_head, dropout=dropout))
            )

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)

        return self.norm(x)  #shape=(b, group, h, w)
    
class totalmodel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, patch_size, num_groups, heads, dim_head, wt_levels=3, wt_type='haar', dropout=0.1, groupsize=4):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels,
        self.num_classes=num_classes
        self.patch_size=patch_size
        self.wt_levels=wt_levels
        self.wt_type=wt_type
        self.num_groups=num_groups     #GSC_kan中卷积层的分组数
        self.heads=heads               #多头注意力机制的头数
        self.dim_head=dim_head         #多头注意力机制中每个头的维度
        self.dropout=dropout
        self.groupsize=groupsize
        
        self.a=nn.Parameter(torch.rand(1, requires_grad=True))   
        self.dp = nn.Dropout(p = dropout)
     
        self.gsf=groupspectralfusion( in_channels, out_channels, patch_size)
       
        self.wavelt_spectial=wavelettransformer(out_channels, wt_levels=wt_levels, wt_type=wt_type)
        
        self.gsc_kan=GSC_kan(out_channels, out_channels, num_groups)
       
        #self.transformer_kan=Transformer_kan(out_channels, patch_size, dim_head=dim_head, heads=heads, dropout=dropout, norm_output=True, groupsize=groupsize)
        self.transformer_kan=GSSA_kan(out_channels, patch_size=patch_size, group_spatial_size=groupsize, heads=heads, dim_head=dim_head, dropout=dropout)
       
        self.bn=nn.BatchNorm2d(out_channels)
      
          
        self.act=nn.GELU()
       
        self.convkan=ConvKAN(out_channels,out_channels,1)
        #self.convkan=nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        self.kan_mlp=nn.Sequential(Reduce('b d h w -> b d', 'mean'),
                                    nn.BatchNorm1d(out_channels),  
                                    Fast_KANLinear(out_channels, num_classes))
        
    def forward(self,x):
        x=self.dp(self.gsf(x))
        
        #fenzhi1
        x_spectial=self.wavelt_spectial(x)
        
        #fenzhi2_1
        x_spatial = self.gsc_kan(x)
        y = x_spatial
        x_spatial = self.transformer_kan(x_spatial)
        x_spatial = self.convkan(x_spatial) + y
        x_spatial = self.act(self.bn(x_spatial))
        
        x_total=self.a*x_spectial+(1-self.a)*x_spatial
     
        x_total=self.kan_mlp(x_total)
        return x_total



if __name__ == '__main__':
    model = totalmodel(in_channels=56, out_channels=28, num_classes=15, patch_size=12, num_groups=4, heads=7, dim_head=4, wt_levels=3, wt_type='haar', dropout=0.1,groupsize=4)
    from torchsummary import summary
    device = torch.device("cuda")
    summary(model.to(device),(56, 12, 12))   
