import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

#Edge-Guided Attention Module
class EGA(nn.Module):
    def __init__(self, in_channels, n_class):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        
        self.conv2 = nn.Conv2d(in_channels, n_class, 1)
        self.output_distance_conv = nn.Conv2d(in_channels, 1, 1)
        self.output_boundary_conv = nn.Conv2d(in_channels, 1, 1)
        
        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x):
        residual = x
        B, C, H, W = x.shape
        xsize = x.size()[2:]   #8, 384, 32, 32
        pred = self.conv2(x)
        distance = self.output_distance_conv(x) #* torch.sigmoid(pred)
        boundary = self.output_boundary_conv(x) #* torch.sigmoid(distance)

        #reverse attention 
        background_att = 1 - torch.sigmoid(pred)
        background_att = background_att.repeat(1, C//3, 1, 1)  # 128 * 3 = 384
        background_x= x * background_att   #8, 384, 32, 32
        #print('boundary.shape:', boundary.shape)
        
        #boudary attention
        pred_feature = x * boundary   #8, 384, 32, 32
        #print('pred_feature.shape:', pred_feature.shape)

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input   #8, 384, 32, 32
        #print('input_feature.shape:', input_feature.shape)

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        #print('attention_map.shape:', attention_map.shape)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        
        
        #out = self.cbam(out)
        return out

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale,dim, n_class):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
        self.convout = nn.Conv2d(dim, n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, feats):
        """Forward function."""
        probs = self.convout(feats)
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class DPGHead(nn.Module):
    def __init__(self, in_ch, mid_ch, pool, fusions):
        super(DPGHead, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = in_ch
        self.planes = mid_ch
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            #[N, D, C, 1]
            input_x = x
            input_x = input_x.view(batch, channel, height*width) # [N, D, C]
            input_x = input_x.unsqueeze(1) # [N, 1, D, C]

            context_mask = self.conv_mask(x) # [N, 1, C, 1]
            context_mask = context_mask.view(batch, 1, height*width) # [N, 1, C]
            context_mask = self.softmax(context_mask) # [N, 1, C]
            context_mask = context_mask.unsqueeze(3) # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)# [N, 1, D, 1]
            context = context.view(batch, channel, 1, 1) # [N, D, 1, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x, y):
        # [N, C, 1, 1]
        context = self.spatial_pool(y)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))# [N, D, 1, 1]
            out = x * channel_mul_term # [N, D, H, W]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)# [N, D, 1, 1]
            out = out + channel_add_term

        return out


class CascadedSubDecoderBinary(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 scale_factor,
                 interpolation_mode='bilinear'):
        super(CascadedSubDecoderBinary, self).__init__()

        self.output_map_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_distance_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.output_boundary_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=interpolation_mode, align_corners=True)


    def forward(self, x):
        map = self.output_map_conv(x) # B, 1, H, W
        distance = self.output_distance_conv(x) * torch.sigmoid(map)
        boundary = self.output_boundary_conv(x) * torch.sigmoid(distance)

        boundary = self.upsample(boundary)
        distance = self.upsample(distance) + torch.sigmoid(boundary)
        map = self.upsample(map) + torch.sigmoid(distance)

        return map, distance, boundary
