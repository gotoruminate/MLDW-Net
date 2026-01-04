import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import torch.nn.init as init
import torch.nn as nn
from torch.nn import Module,Conv3d, Parameter,Softmax

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim,Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(LayerNormalize(dim,MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

NUM_CLASS = 5

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, channle = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)
        out = self.gamma*out + x
        #print('out', out.shape)
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)
        out = self.gamma*out + x  #C*H*W
        #print('out', out.shape)
        return out

class ViT(nn.Module):
    def __init__(self, *,in_channels=1,image_size = 5, patch_size = 1, num_classes = 5, dim = 32, depth = 2, heads = 8,
                 mlp_dim = 256, channels = 32, dim_head = 8, dropout = 0.3, emb_dropout = 0.3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(64,dim)#patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.attention_spectral = CAM_Module(10)
        self.attention_spatial = PAM_Module(32)

        self.conv2d_f = nn.Sequential(
            nn.Conv2d(in_channels=432,out_channels=32,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv3d_f = nn.Sequential(
            nn.Conv3d(in_channels,out_channels=16,kernel_size=(3,3,3),padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(in_channels=33, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=64 * 27, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_features_1 = nn.Sequential(
            nn.Conv2d(in_channels=1792, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pro_head = nn.Linear(dim, channels, nn.ReLU())
    def forward(self, img, mask = None, mode='test'):
        p = self.patch_size
        #print(img.shape)#torch.Size([64, 27, 5, 5])
        #img = img.unsqueeze(1)
        #分支一
        #卷积+串行
        x1 = self.conv3d_f(img) #torch.Size([64, 16, 27, 5, 5])
        #print(x1.shape)
        x1 = self.attention_spectral(x1) #torch.Size([64, 16, 27, 5, 5])
        #print(x1.shape)
        x1 = rearrange(x1,'b c l h w -> b (l c) h w') #torch.Size([64, 432, 5, 5])
        #print(x1.shape)
        x1 = self.conv2d_f(x1)#torch.Size([64, 32, 5, 5])
        #print(x1.shape)
        x1 = (torch.unsqueeze(x1,0)).permute([1,2,0,3,4])#torch.Size([64, 32, 1, 5, 5])
        #print(x1.shape)
        x1 = self.attention_spatial(x1)#torch.Size([64, 32, 1, 5, 5])
        #print(x1.shape)
        x1 = rearrange(x1,'b c l h w -> b (c l) h w')
        #print(x1.shape)#torch.Size([64, 32, 5, 5])
        #分支二
        res = img
        x2 = self.conv3d_features(img)#torch.Size([64, 32, 27, 5, 5])
        #print(x2.shape)
        x2 = torch.cat((x2, res), dim=1)#torch.Size([64, 33, 27, 5, 5])
        #print(x2.shape)
        x2 = self.conv3d_features_1(x2)#torch.Size([64, 64, 27, 5, 5])
        #print(x2.shape)
        res1 = rearrange(x2, 'b c h w y -> b (c h) w y')#torch.Size([64, 64, 27, 5, 5])
        #print(x2.shape)
        x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2], x2.shape[3], x2.shape[4])#torch.Size([64, 1728, 5, 5])
        #print(x2.shape)
        x2 = self.conv2d_features(x2)#torch.Size([64, 64, 5, 5])
        #print(x2.shape)
        x2 = torch.cat((x2, res1), dim=1)#torch.Size([64, 704, 5, 5])
        #print(x2.shape)
        x2 = self.conv2d_features_1(x2)#torch.Size([64, 32, 5, 5])
        #print(x2.shape)
        img = torch.cat([x1,x2],dim=1)#torch.Size([64, 64, 5, 5])
        #print(img.shape)
        #print("img shape：",img.shape)#img shape： torch.Size([64, 1, 174, 13, 13])
        #img = img.reshape(img.shape[0],img.shape[1]*img.shape[2],img.shape[3],img.shape[4])
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)#torch.Size([64, 25, 52])
        #print(x.shape)
        #print("before x:",x.size())
        x = self.patch_to_embedding(x)#torch.Size([64, 25, 1024])
        #print(x.shape)
        #print("after x:",x.size())
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)#torch.Size([64, 1, 1024])
        #print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)#torch.Size([64, 26, 1024])
        #print(x.shape)
        x += self.pos_embedding[:, :(n + 1)]#torch.Size([64, 26, 1024])
        #print(x.shape)
        x = self.dropout(x)#torch.Size([64, 26, 1024])
        #print(x.shape)
        x = self.transformer(x, mask)#torch.Size([64, 26, 1024])
        #print(x.shape)
        x = self.to_cls_token(x[:, 0])#
        #print(x.shape)
        if mode == 'test':
            clss = self.mlp_head(x)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(x))
            clss = self.mlp_head(x)
            return clss, proj
if __name__ == '__main__':
    model = ViT(image_size = 13,patch_size = 1,num_classes = NUM_CLASS,dim = 1024,depth = 2,
              heads = 16,mlp_dim = 2048,channels =10,dropout = 0.1,emb_dropout = 0.1)
    model.eval()
    #print(model)
    input_1 = torch.randn(64, 27, 5, 5)
    #input_2 = torch.randn(64, 1, 10, 5, 5)
    out1,out2=model(input_1,mode = 'train')
    print(out1.size(),out2.size())
    #x_feature,x1,x,output_x,y_feature,y1,y,output_y = model(input_1,input_2)
    #summary(model)