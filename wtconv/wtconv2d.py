import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .util import wavelet


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )                                                        #padding='same'保证特征图大小不变，前提步长为1
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
         
        curr_x_ll = x  # 初始化当前的低频分量为输入数据
        # 多级小波分解过程
        for i in range(self.wt_levels):
             # 记录当前低频分量的形状
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # 如果当前低频分量的高度或宽度为奇数，则进行零填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)    #四个元素对应右、左、下、上四个方向的填充数
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            # 将当前低频分量进行小波变换并提取低频分量
            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  #将高频分量展平，（b, c, 4, h // 2, w // 2）→（b，c×4，h // 2, w // 2）
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)) #先卷积在缩放   对四个分量都进行卷积和缩放
            curr_x_tag = curr_x_tag.reshape(shape_x)     #（b，c×4，h // 2, w // 2）→（b, c, 4, h // 2, w // 2）

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])     #对应YLL
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])    #对应YH
        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):   #反向range
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)       #对应Z(i+1)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter) #对应Z(i)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):                          
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
#dims=[1,in_channels*4,1,1]，所以在mul时使用了广播机制，这里相当于通道缩放：不同通道间的缩放因子不同和批次和空间位置无关。