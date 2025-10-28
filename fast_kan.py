import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.01, **kw) -> None:     
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):                 #径向基函数
    def __init__(
        self,
        grid_min: float ,
        grid_max: float ,
        num_grids: int ,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=True)                         
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)       

    def forward(self, x):                        
        x_out=torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
        return x_out.to(torch.float32)     

class Fast_KANLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -1.,
        grid_max: float = 1.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
    
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.a=nn.Parameter(torch.rand(1, requires_grad=True))
        
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)
            

    def forward(self, x):                                       #分组的输入shape=[（batch_size）*L, （in_channels//self.num_groups） * kernel_size * kernel_size]
       
        mins,_ = torch.min(x, dim=1, keepdim=True)
        maxs,_ = torch.max(x, dim=1, keepdim=True)
        # 防止除零
        scale = maxs - mins
        scale[scale < 1e-8] = 1.0
        # 缩放至 [-1, 1]
        scaled_x= 2*(x - mins) / scale - 1 
        
        spline_basis = self.rbf(scaled_x)
        ret = self.spline_linear(spline_basis.reshape(*spline_basis.shape[:-2], -1))     #在RBF中x[..., None]将x扩充了1个维度，所以这里将最后两个维度合并:input_dim * num_grids
        if self.use_base_update:
            base = self.base_linear(self.base_activation(scaled_x))
            ret = self.a*ret + (1-self.a)*base
        return ret     #分组的输出shape=(（batch_size）*L, output_dim//self.num_groups)
   
class ConvKAN1d(nn.Module):
    def __init__(
        self,
        
        dim: int,
        num: int ,             
        grid_min: float = -1.,
        grid_max: float = 1.,
        num_grids: int = 8,
        base_activation = F.silu,
       
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.num = num
        
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)    
        
        self.a=nn.Parameter(torch.rand(1, requires_grad=True))
        self.weight = nn.Parameter(torch.rand(num_grids, requires_grad=True))  
        
        self.base_activation = base_activation
        self.base_linear = nn.Linear(dim, dim)
        
#数据输入形状 b*h*w, dim, 1+g1*g2
    def forward(self, x):
        
        mins ,_= torch.min(x, dim=2, keepdim=True)
        maxs ,_= torch.max(x, dim=2, keepdim=True)
        # 防止除零
        scale = maxs - mins
        scale[scale < 1e-8] = 1.0
        # 缩放至 [-1, 1]
        scaled_x= 2 * (x - mins) / scale - 1       
  
        scaled_x = scaled_x.repeat_interleave(self.num, dim=1)        #形状 b*h*w, num*dim, 1+g1*g2
        
        spline_basis = self.rbf(scaled_x)          #形状 b*h*w, num*dim, 1+g1*g2，8
        weight = self.weight.view(1, 1, 1, -1)   
        spline_basis = spline_basis * weight  
        ret = torch.sum(spline_basis, dim=-1)    #形状 b*h*w, num*dim, 1+g1*g2
        
        base = self.base_linear(self.base_activation(scaled_x))
        ret = self.a*ret + (1-self.a)*base
        return ret