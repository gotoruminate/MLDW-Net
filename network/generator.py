import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *

class SpaRandomization(nn.Module):# 空间随机化
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False) #特征图归一化
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device) #可学习的参数

    def forward(self, x,):
        N, C, H, W = x.size() #张量维度
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1) # x(N,C,H*W)
            mean = x.mean(-1, keepdim=True) #mean(N,C,1)
            var = x.var(-1, keepdim=True) #var(N,C,1)
            
            x = (x - mean) / (var + self.eps).sqrt() #对x做标准化，x(N,C,H*W)
            
            idx_swap = torch.randperm(N) #生成o-N-1的随机排列
            alpha = torch.rand(N, 1, 1) #alpha(N,1,1)
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap] #更新均值 
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap] #代码使用方差而论文标注标准差 更新方差

            x = x * (var + self.eps).sqrt() + mean #与论文中的公式不同 恢复原来的标准化
            x = x.view(N, C, H, W) #恢复原来的形状 x(N,C,H,W)

        return x, idx_swap #返回x和随机排列idx_swap


class SpeRandomization(nn.Module): #光谱随机化
    def __init__(self,num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False) #特征图实例归一化

    def forward(self, x, idx_swap,y=None):
        N, C, H, W = x.size() #获取x的形状

        if self.training:
            x = x.view(N, C, -1) #x(N,C,H*W)
            mean = x.mean(1, keepdim=True) #mean(N,1,H*W)
            var = x.var(1, keepdim=True) #var(N,1,H*W)
            
            x = (x - mean) / (var + self.eps).sqrt() #标准化
            if y!= None:
                for i in range(len(y.unique())):
                    index= y==y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach() #x(N,C,H*W)

                x = x * (var + self.eps).sqrt() + mean #恢复原来的标准化
            x = x.view(N, C, H, W) #恢复原来的形状
        return x


class AdaIN2d(nn.Module): #自适应实例归一化
    def __init__(self, style_dim, num_features):#style_dim：风格向量维度 num_features:通道数
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False) #对每个通道做实例归一化
        self.fc = nn.Linear(style_dim, num_features*2) #全连接操作
    def forward(self, x, s): #x(N,C,H,W) s(N,style_dim)
        h = self.fc(s) # 将风格向量映射到归一化参数 h(N,2*C)
        h = h.view(h.size(0), h.size(1), 1, 1) #h(N,2*C,1,1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1) #gamma(N,C,1,1) beta(N,C,1,1)
        return (1 + gamma) * self.norm(x) + beta #1+gamma:缩放因子 beta:偏移操作 gamma(N,C,H,W) beta(N,C,H,W)
        #return (1+gamma)*(x)+beta

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape) #x(N,C*H*W)

class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[5, 5], zdim=10, device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2 #步长为1
        self.zdim = zdim # 风格向量维度设置为10
        self.imdim = imdim # 输入通道维数
        self.imsize = imsize #图片尺寸【13，13】
        self.device = device #设备配置
        num_morph = 4 #设置MorphNet输出通道数
        self.Morphology = MorphNet(imdim) #设置MorphNet输入通道为imdim
        self.adain2_morph = AdaIN2d(zdim, num_morph) #风格映射

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1) #输入通道imdim=3，输出通道3，卷积核1，步长1
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1) #输入通道3，输出通道n=16，卷积核1，步长1
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1) #输入通道imdim=3，输出通道n=16，卷积核大小13，步长1
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0]) #输入通道n=16，输出通道n=16，卷积核大小13
        self.conv1 = nn.Conv2d(n+n+num_morph, n, kernelsize, 1, stride) #输入通道n+n+num_morph=36,输出通道n=16，卷积核大小3 
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride) #输入通道n=16，输出通道imdim=3，卷积核大小3
        self.speRandom = SpeRandomization(n) #num_feature=n=16 
        self.spaRandom = SpaRandomization(3, device=device) #num_feature=3

    def forward(self, x): 

        x_morph= self.Morphology(x) #
        z = torch.randn(len(x), self.zdim).to(self.device)
        x_morph = self.adain2_morph(x_morph, z)

        x_spa = F.relu(self.conv_spa1(x))
        x_spe = F.relu(self.conv_spe1(x))
        x_spa, idx_swap = self.spaRandom(x_spa)
        x_spe = self.speRandom(x_spe,idx_swap)
        x_spe = self.conv_spe2(x_spe)
        x_spa = self.conv_spa2(x_spa)
        
        x = F.relu(self.conv1(torch.cat((x_spa,x_spe,x_morph),1)))
        x = torch.sigmoid(self.conv2(x))

        return x


