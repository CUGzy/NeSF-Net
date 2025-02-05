import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchvision.transforms.functional import rgb_to_grayscale
from .modules import *
from torch.nn import TransformerEncoderLayer
import math
from .gaps import GlobalAvgPool2D
from .attention import CAM_Module,PAM_Module,CrossAttention
from .FMS import FMS_block

def bchw2bcl(x):
    b,c,h,w = x.shape
    return x.view(b,c,h*w).contiguous()

def bchw2blc(x):
    b,c,h,w = x.shape
    return x.view(b,c,h*w).permute(0,2,1).contiguous()

def bcl2bchw(x):
    b,c,l = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b,c,h,w).contiguous()

def blc2bchw(x):
    b,l,c = x.shape
    h = int(math.sqrt(l))
    w = h
    return x.view(b,h,w,c).permute(0,3,1,2).contiguous()


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, sp=True, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        self.sp = sp

        if  self.sp: # for 512x512 input
            self.sfs = nn.ModuleList(
                [FMS_block(in_ch=96, out_ch=192),
                FMS_block(in_ch=192, out_ch=384),
                FMS_block(in_ch=384, out_ch=768)]
            )
            
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            #nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        global_sp = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            '''
            if i != 3:
                x_g = self.sfs[i](x)
                global_sp.append(x_g)
                #print(f'第{i}个x_g.shape:', x_g.shape)
            if i != 0 and self.sp:
                x = x + global_sp[i-1]
            '''
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}


@register_model
def convnext_encoder(pretrained=False,in_22k=False,sp=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], sp=sp,**kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvChannelEmbeding(nn.Module):
    def __init__(self,in_c,down_ratio):
        super(ConvChannelEmbeding,self).__init__()

        self.in_c = in_c 
        self.channel_embeding = nn.Sequential(
            nn.Conv2d(in_c,in_c,down_ratio,down_ratio,0,groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU6()
        )

    def forward(self,x):
        """
        torch.Size([1, 64, 256, 256]) x1    32
        torch.Size([1, 64, 128, 128]) x2    16   
        torch.Size([1, 128, 64, 64]) x3     8
        torch.Size([1, 256, 32, 32]) x4     4
        torch.Size([1, 512, 16, 16]) x5      
        """
              
        x = self.channel_embeding(x)  
        return x        


class ChannelAttnBlock(nn.Module):
    def __init__(self,in_c,down_ratio,h,heads):
        super(ChannelAttnBlock,self).__init__()

        self.in_c = in_c
        self.down_ratio = down_ratio
        self.dim = int((h//down_ratio)*(h//down_ratio))
        self.heads = heads
        #self.num_layers = num_layers

        self.ce = ConvChannelEmbeding(self.in_c,self.down_ratio)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.in_c, self.dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.attn = nn.Sequential(
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4),
            TransformerEncoderLayer(self.dim,self.heads,self.dim*4)
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=self.down_ratio)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self,x):
        x = self.ce(x)      # torch.Size([2, 64, 16, 16])
        shortcut = x

        x = bchw2bcl(x)     # torch.Size([2, 64, 256])

        x = self.pos_drop(x + self.pos_embed)
        x = self.attn(x)    # torch.Size([2, 64, 256])
        x = bcl2bchw(x) + shortcut    # torch.Size([2, 64, 16, 16])
        x = self.up(x)      # torch.Size([2, 64, 256, 256])
        return x       


class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, in_c, init_value=1e-2):
        super().__init__()
        self.inChannels = in_c
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((in_c)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            return scale * x


class DWConv3x3(nn.Module):
    def __init__(self, in_c):
        super(DWConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True, groups=in_c)

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_c, out_c, hid_c, ls=1e-2,drop=0.0):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_c)

        self.fc1 = nn.Conv2d(in_c, hid_c, 1)
        self.dwconv = DWConv3x3(hid_c)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_c, out_c, 1)

        self.layer_scale = LayerScale(in_c, init_value=ls)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        
        shortcut = x.clone()

        # ffn
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        # layer scale
        x = self.layer_scale(x)
        x = self.drop(x)

        out = shortcut + x
        return out


class LocalConvAttention(nn.Module):

    def __init__(self, dim):
        super(LocalConvAttention, self).__init__()
        
        # aggression local info
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) 

        self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        shortcut = x.clone()
        
        x_33 = self.conv0(x)

        x_15 = self.conv0_1(x)
        x_15 = self.conv0_2(x_15)

        x_111 = self.conv1_1(x)
        x_111 = self.conv1_2(x_111)

        add = x_33 + x_15 + x_111
        
        mixer = self.conv3(add)
        out = mixer * shortcut

        return out


class GlobalSelfAttentionV3(nn.Module):
    def __init__(self, dim, h ,drop=0.0):
        super(GlobalSelfAttentionV3, self).__init__()
        
        # aggression local info
        self.local_embed = nn.Sequential(
            nn.Conv2d(dim, dim//4, 4, 4 , groups=dim//4),
            nn.BatchNorm2d(dim//4),
            nn.ReLU6())

        self.dim = dim//4
        self.real_h = int(h//4)
        self.window_size = [self.real_h,self.real_h]
        
        if self.dim <=64:
            self.num_heads = 2
        else:
            self.num_heads = 4

        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(drop)

        self.conv_out = nn.Conv2d(self.dim, dim, 1, 1 )
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
                
        # local embed
        x = self.local_embed(x) # b c h w
        b,c,h,w = x.shape  
        x = x.view(b,c,h*w).permute(0,2,1).contiguous() # blc torch.Size([1, 256, 64])

        # self-attn
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        x = blc2bchw(x)
        out = self.up(self.conv_out(x))
       
        return out


class SpatialFormer(nn.Module):
    def __init__(self, dim, h, ls=1e-2, drop=0.0,):
        super(SpatialFormer,self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.detail_attn = LocalConvAttention(dim)
        self.global_attn = GlobalSelfAttentionV3(dim,h)

        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls)
        self.drop = nn.Dropout(p=drop)

        hidden_dim = 4*dim
        self.ffn = FFN(in_c=dim, out_c=dim, hid_c=hidden_dim, ls=ls, drop=drop)
        
    def forward(self, x):

        shortcut = x.clone()

        # proj1
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        
        # attn    
        xd = self.detail_attn(x)
        xg = self.global_attn(x)
        attn = xd + xg

        # proj2
        attn = self.proj2(attn)
        attn = self.layer_scale(attn)
        attn = self.drop(attn)

        attn_out = attn + shortcut

        # ffn
        out = self.ffn(attn_out)

        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False,padding=0):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=3,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        #c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        c2wh = dict([(64, 112), (96, 56), (192, 28), (384, 14), (768, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        #self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        #self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            #print('scale_idx:', scale_idx)
            #print('feature.shape:', feature.shape)
            #print('spatial_attention_map.shape:', spatial_attention_map.shape)

            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha + feature * spatial_attention_map * self.beta)
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_branches=3,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8):
        super(UpsampleBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        self.attention_layer = MFMSAttentionBlock(out_channels, scale_branches, frequency_branches, 
                                                frequency_selection, block_repetition, min_channel, min_resolution)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention_layer(x)
        #x = self.conv2(x)

        return x

###################类内与类间增强
def one_hot(ori, classes):
    batch, h, w = ori.size()
    new_gd = torch.zeros((batch, classes, h, w), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = torch.nonzero((ori == j))
        for i in range(len(index_list)):
            batch, height, width = index_list[i]
            new_gd[batch, j, height, width] = 1

    return new_gd.float()

def both(ori, attention,classes):
    batch, channel, h, w = ori.size()
    ori_out = torch.zeros(size=(ori.size()), dtype=ori.dtype).cuda()   #8,768,16,16
    intar_class = torch.zeros(size=(batch, channel, classes), dtype=ori.dtype).cuda()   #8,768,3
    attention = torch.argmax(attention, dim=1)  # (B, H, W)   #8,16,16
    attention = one_hot(attention, classes=classes)  # (B, Cl, H, W)   #8,3,16,16

    # intra class
    for category in range(attention.shape[1]):
        category_map = attention[:, category:category + 1, ...]  # (B, 1, H, W) 8,1,16,16
        ori_category = torch.einsum('bchw, bfhw -> bchw', ori, category_map)   #8,768,16,16

        sum_category = torch.sum(ori_category, dim=(2, 3))  # (B, C)  8,768
        number_category = torch.sum(category_map, dim=(2, 3)) + 1e-5  # (B, 1)  8,1
        avg_category = sum_category / number_category  # (B, C)   8,768
        intar_class[:, :, category] = avg_category  # (B, C, Cl)   8,768,3

        avg_category2 = torch.einsum('bc, bfhw -> bchw', avg_category, category_map)   #8,768,16,16
        ori_out = ori_out + avg_category2

    # inter class
    D = 0
    
    for i in range(classes):
        for j in range(i + 1, classes):
            D += torch.sum(torch.norm(intar_class[:, :, i] - intar_class[:, :, j], dim=1))
    
    Euclidean_dis = D
    '''
    A = B = intar_class  # (B, C, Cl)   8,768,3
    BT = B.permute(0, 2, 1)  # (B, Cl, C)   8,3,768
    vecProd = torch.matmul(BT, A)  # (B, Cl, Cl)   8,3,3
    SqA = A ** 2   #8,768,3
    sumSqA = torch.sum(SqA, dim=1).unsqueeze(dim=2).repeat(1, 1, 3)  # (B, Cl, Cl)   8,3,3
    Euclidean_dis = sumSqA * 2 - vecProd * 2   #8,3,3
    Euclidean_dis = torch.sum(Euclidean_dis, dim=(1, 2))   #8
    Euclidean_dis = torch.pow(Euclidean_dis, 1 / 2).sum()   #[]
    '''
    return ori_out, Euclidean_dis


###################八方向特征
class SDE_module(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(SDE_module, self).__init__()
        self.inter_channels = in_channels // 8

        self.att1 = DANetHead(self.inter_channels,self.inter_channels)
        self.att2 = DANetHead(self.inter_channels,self.inter_channels)
        self.att3 = DANetHead(self.inter_channels,self.inter_channels)
        self.att4 = DANetHead(self.inter_channels,self.inter_channels)
        self.att5 = DANetHead(self.inter_channels,self.inter_channels)
        self.att6 = DANetHead(self.inter_channels,self.inter_channels)
        self.att7 = DANetHead(self.inter_channels,self.inter_channels)
        self.att8 = DANetHead(self.inter_channels,self.inter_channels)


        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        #self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))
        
        if num_class<32:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, num_class*8, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_class*8, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, in_channels, 1),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, x, d_prior):

        enc_feat = self.reencoder(d_prior)   
        feat1 = self.att1(x[:,:self.inter_channels], enc_feat[:,0:self.inter_channels])
        feat2 = self.att2(x[:,self.inter_channels:2*self.inter_channels],enc_feat[:,self.inter_channels:2*self.inter_channels])
        feat3 = self.att3(x[:,2*self.inter_channels:3*self.inter_channels],enc_feat[:,2*self.inter_channels:3*self.inter_channels])
        feat4 = self.att4(x[:,3*self.inter_channels:4*self.inter_channels],enc_feat[:,3*self.inter_channels:4*self.inter_channels])
        feat5 = self.att5(x[:,4*self.inter_channels:5*self.inter_channels],enc_feat[:,4*self.inter_channels:5*self.inter_channels])
        feat6 = self.att6(x[:,5*self.inter_channels:6*self.inter_channels],enc_feat[:,5*self.inter_channels:6*self.inter_channels])
        feat7 = self.att7(x[:,6*self.inter_channels:7*self.inter_channels],enc_feat[:,6*self.inter_channels:7*self.inter_channels])
        feat8 = self.att8(x[:,7*self.inter_channels:8*self.inter_channels],enc_feat[:,7*self.inter_channels:8*self.inter_channels])
        
        feat = torch.cat([feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8],dim=1)

        sasc_output = self.final_conv(feat)
        #sasc_output = sasc_output+x

        return sasc_output

class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))
        
        self.attn = CrossAttention(in_channels, num_heads=4)
        self.global_local = FMS_block(in_ch=in_channels, out_ch=inter_channels)

    def forward(self, x, enc_feat):
        #x.shape: torch.Size([10, 96, 16, 16])
        #enc_feat.shape: torch.Size([10, 96, 1, 1])

        B, C, H, W = enc_feat.shape
        att_enc_feat = self.attn(x, enc_feat)
        att_enc_feat = att_enc_feat.permute(0, 2, 1).view(B, C, 1, 1)
        feat_sum = self.global_local(x)
        
        '''
        feat1 = self.conv5a(x)   #b,96,16,16
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        
        feat2 = self.conv5c(x)   #b,96,16,16
        #print('feat2.shape:', feat2.shape)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv+sc_conv
        '''
        feat_sum = feat_sum*F.sigmoid(att_enc_feat)
        feat_sum = self.conv8(feat_sum)
        

        return feat_sum

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Model(nn.Module):
    def __init__(self, 
                 n_class=2, 
                 pretrained = True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        
        self.backbone = convnext_encoder(pretrained,in_22k=True,sp=False)

        self.Up5 = up_conv(ch_in=768, ch_out=384)
        self.decoder_stage5 = UpsampleBlock(in_channels=768, out_channels=384)

        self.Up4 = up_conv(ch_in=384, ch_out=192)
        self.decoder_stage4 = UpsampleBlock(in_channels=384, out_channels=192)

        self.Up3 = up_conv(ch_in=192, ch_out=96)
        self.decoder_stage3 = UpsampleBlock(in_channels=192, out_channels=96)
        
        #self.Up2 = up_conv(ch_in=96, ch_out=96)
        #self.decoder_stage2 = UpsampleBlock(in_channels=96, out_channels=64)
        
        self.Up4x = LRDU(96,4)      
        self.convout = nn.Conv2d(96, n_class, kernel_size=1, stride=1, padding=0)
        
        ###---------global&local---------
        self.fuseFeature_1 = FMS_block(in_ch=384, out_ch=192)
        
        ###---------八方向特征--------
        self.gap = GlobalAvgPool2D()
        out_planes = n_class*8
        self.channel_mapping = nn.Sequential(
                    nn.Conv2d(768, out_planes, 3,1,1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(True)
                )
        self.direc_reencode = nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, 1),
                    # nn.BatchNorm2d(out_planes),
                    # nn.ReLU(True)
                )
        self.sde_module = SDE_module(768,768,out_planes)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x128,x64,x32,x16 = self.backbone(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """
        
        #### directional Prior ####
        directional_c5 = self.channel_mapping(x16)
        mapped_c5=F.interpolate(directional_c5,scale_factor=32,mode='bilinear',align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)
        d_prior = self.gap(mapped_c5)   # torch.Size([10, 24, 1, 1])
        #print('d_prior.shape:', d_prior.shape)
        x16 = self.sde_module(x16,d_prior)   #torch.Size([8, 768, 16, 16])
        
        d32 = self.Up5(x16)
        d32 = torch.cat([x32,d32],dim=1)
        d32 = self.decoder_stage5(d32)

        d64 = self.Up4(d32)     
        d64 = torch.cat([x64,d64],dim=1)
        d64 = self.decoder_stage4(d64)

        d128 = self.Up3(d64)    
        d128 = torch.cat([x128,d128],dim=1)
        d128 = self.decoder_stage3(d128)
        
        #d256 = self.Up2(d128)    
        #d256 = self.decoder_stage2(d256)

        d2 = self.Up4x(d128)
        d1 = self.convout(d2)
        
        #return [d1,attention,Euclidean_dis]
        return [d1,mapped_c5]


if __name__ == "__main__":

    model = Model(2,False)
    img = torch.rand((1,3,512,512))
    output = model(img)
    print(output.shape)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 

