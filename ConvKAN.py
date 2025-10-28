import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_kan import Fast_KANLinear

class groupConvKAN(nn.Module):
    
    def __init__(self, 
                in_channels, 
                out_channels,
                num_groups, 
                kernel_size, 
                stride=1, 
                padding=0, 
                grid_size=8,              
                scale_spline=1.0,           
                base_activation=torch.nn.SiLU(),           
                grid_range=[-1, 1],
                version= "Fast",
                ):
        super(groupConvKAN, self).__init__()

        self.version = version
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #将卷积核展开成一个向量，这样就可以用线性层来处理。卷积神经网络中的卷积操作的底层实现。
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)         #kernel_size是一个整数，表示卷积核高宽相等

 
        self.linear = nn.ModuleList([
            Fast_KANLinear(
                input_dim = in_channels//self.num_groups* self.kernel_size * self.kernel_size,
                output_dim = out_channels//self.num_groups,
                num_grids=grid_size,
                spline_weight_init_scale=scale_spline,
                base_activation=base_activation,
                grid_min = grid_range[0],
                grid_max = grid_range[1],
                )  for _ in range(num_groups)]
                )
   
    def forward(self, x):  

        batch_size, in_channels, height, width = x.size()
    
        assert x.dim() == 4
        
        assert in_channels//self.num_groups == in_channels/self.num_groups   #保证能整除 //是整除取整数部分 /是普通除法，返回浮点数
        
        #实现KAN的分组卷积
        group_x = []
        x=torch.reshape(x, (batch_size, self.num_groups, in_channels//self.num_groups, height, width))
        for i in range(self.num_groups):
            x_group = x[:,i,:,:,:]      
            patches = self.unfold(x_group)      #shape=(batch_size, in_channels//self.num_groups * kernel_size * kernel_size, L) L是局部块的个数
            patches = patches.transpose(1, 2)  
            patches = patches.reshape(-1, in_channels//self.num_groups* self.kernel_size * self.kernel_size)    # [样本个数, 输入特征数]
            out = self.linear[i](patches)              #shape=(batch_size*L, output_dim//self.num_groups)
                      
            group_x.append(out)
        out = torch.cat(group_x, dim=1)   #shape=(batch_size*L, output_dim)  L是局部块的个数
        out = out.reshape(batch_size, -1, out.size(-1))             #shape=(batch_size, L, output_dim)
 
        # Calculate the height and width of the output.
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        

        out = out.transpose(1, 2) 
        out = out.reshape(batch_size, self.out_channels, out_height, out_width) 
        
        return out
    #非分组kan卷积
class ConvKAN(nn.Module):
    
    def __init__(self, 
                in_channels, 
                out_channels,               
                kernel_size, 
                stride=1, 
                padding=0, 
                grid_size=8,              
                scale_spline=1.0,           
                base_activation=torch.nn.SiLU(),           
                grid_range=[-1, 1],
                
                ):
        super(ConvKAN, self).__init__()

        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #将卷积核展开成一个向量，这样就可以用线性层来处理。卷积神经网络中的卷积操作的底层实现。
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)         #kernel_size是一个整数，表示卷积核高宽相等

        self.linear = Fast_KANLinear(
                input_dim = in_channels* self.kernel_size * self.kernel_size,
                output_dim = out_channels,
                num_grids=grid_size,
                spline_weight_init_scale=scale_spline,
                base_activation=base_activation,
                grid_min = grid_range[0],
                grid_max = grid_range[1],
        ) 
    
    def forward(self, x):  

        batch_size, in_channels, height, width = x.size()
        
        assert x.dim() == 4
        assert in_channels == self.in_channels

        patches = self.unfold(x)      # [batch_size, in_channels * kernel_size * kernel_size,L] L 是局部块的个数
        patches = patches.transpose(1, 2) 
        patches = patches.reshape(-1, in_channels * self.kernel_size * self.kernel_size)    # [样本个数, 输入特征数]

        out = self.linear(patches)
        
        out = out.reshape(batch_size, -1, out.size(-1))  
        # Calculate the height and width of the output.
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, self.out_channels, out_height, out_width) 
        
        return out