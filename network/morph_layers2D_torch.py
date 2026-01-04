import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Morphology(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels #输入通道
        self.out_channels = out_channels #输出通道
        self.kernel_size = kernel_size #核大小
        self.soft_max = soft_max #是否使用软最大化
        self.beta = beta #softmax中的beta参数
        self.type = type

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True) #创建1张量
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1) #展开操作

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)
        
        # unfold
        x = self.unfold(x)  # x(B, Cin*kH*kW, L), L是小块的数量
        x = x.unsqueeze(1)  # x(B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L)) # 将L的平方根作为输出的高度和宽度

        # erosion 对卷积核进行展平
        weight = self.weight.view(self.out_channels, -1) # weight(Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # weight(1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # x(B, Cout, Cin*kH*kW, L) 卷积核与图像特征相减
        elif self.type == 'dilation2d':
            x = weight + x # x(B, Cout, Cin*kH*kW, L) 卷积核与图像特征相加
        else:
            raise ValueError
        
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L) 如果不使用softmax，进行最大池化
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L) # 使用softmax进行平滑处理

        if self.type == 'erosion2d': #如果是腐蚀操作，结果需要取反
            x = -1 * x

        # instead of fold, we use view to avoid copy 将展开的特征重塑为图像的形状
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x 

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1) # 计算有效的卷积核大小
    pad_total = kernel_size_effective - 1 # 总的填充量
    pad_beg = pad_total // 2 # 开始部分的填充量
    pad_end = pad_total - pad_beg # 结束部分的填充量
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end)) # 对输入进行填充
    return padded_inputs # 返回填充后的输入
