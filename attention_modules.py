"""
ALL ATTENTION MODULES IMPLEMENTED IN THIS PROJECT

original implementation modules can be found in ultralytics/nn/modules/conv.py

all customized model config file can be found in ultralytics/config/model/v8
"""
import torch
from torch import nn as nn

__all__ = (
    "ECAAttention",
    "ShuffleAttention",
    "GAMAttention",
    "CBAM",
    "TripletAttention"
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

#####################################
#    EFFICIENT CHANNEL ATTENTION    #
#####################################

class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
#####################################
#         SHUFFLE ATTENTION         #
#####################################
class ShuffleAttention(nn.Module):

    def __init__(self, c1=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(c1 // (2 * G), c1 // (2 * G))
        self.cweight = nn.parameter.Parameter(torch.zeros(1, c1 // (2 * G), 1, 1))
        self.cbias = nn.parameter.Parameter(torch.ones(1, c1 // (2 * G), 1, 1))
        self.sweight = nn.parameter.Parameter(torch.zeros(1, c1 // (2 * G), 1, 1))
        self.sbias = nn.parameter.Parameter(torch.ones(1, c1 // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()
                    
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

#####################################
#      GLOBAL ATTENTION MODULE      #
#####################################
class GAM_ChannelAttention(nn.Module):
    """ Construct Channel Attention Module for GAM"""

    def __init__(self, channels, rate = 4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // rate),
            nn.ReLU(inplace=True),
            nn.Linear(channels // rate, channels),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):

        b, c, h, w = x.shape
        F1_permuted = x.permute(0, 2, 3, 1).view(b, -1, c)
        F2_permuted = self.channel_attention(F1_permuted).view(b, h, w, c)
        F2 = F2_permuted.permute(0, 3, 1, 2)

        return x * self.act(F2)
    
class GAM_SpatialAttention(nn.Module):
    """ Construct Spatial Attention module for GAM"""
    
    def __init__(self, channels, groups=True, rate=4):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // rate, 7, 1, 3, groups = rate) if groups 
                                    else nn.Conv2d(nn.Conv2d(channels, channels // rate, 7, 1, 3)),
            nn.BatchNorm2d(channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // rate, channels, 7, 1, 3, groups = rate) if groups
                                    else nn.Conv2d(nn.Conv2d(channels, channels // rate, 7, 1, 3)),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.act(self.spatial_attention(x))


class GAMAttention(nn.Module):
    """ Construct Global Attention Mechanism Module"""

    def __init__(self, c1, groups=True, rate=4):
        super().__init__()

        self.channel_attention = GAM_ChannelAttention(c1)

        self.spatial_attention = GAM_SpatialAttention(c1)
    
    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


###########################################
#  CONVOLUTIONAL BLOCK ATTENTION MODULE   #
###########################################
class CBAM_ChannelAttention(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, rate=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // rate, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(channels // rate, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAM_SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = CBAM_ChannelAttention(c1)
        self.spatial_attention = CBAM_SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
    

###########################################
#        TRIPLET ATTENTION MODULE         #
###########################################
class ZPool(nn.Module):
    def forward(self, x):
        "Applies the ZPool."
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, kernel_size = 7):
        super().__init__()
        self.zpool = ZPool()
        self.conv = Conv(2, 1, kernel_size, act = False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.conv(self.zpool(x)))

class TripletAttention(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.ChannelHeight= AttentionGate()
        self.ChannelWidth = AttentionGate()
        self.HeightWidth = AttentionGate()
    
    def forward(self, x):
        x_CW = x.permute(0, 2, 1, 3).contiguous()    #Interaction of Channel and Width
        x_CW = self.ChannelWidth(x_CW)
        x_CW = x_CW.permute(0, 2, 1, 3).contiguous()

        x_CH = x.permute(0, 3, 2, 1).contiguous()    #Interaction of Channel and Height
        x_CH = self.ChannelHeight(x_CH)
        x_CH = x_CH.permute(0, 3, 2, 1).contiguous()

        x_HW = self.HeightWidth(x)

        return 1/3 * (x_CW + x_CH + x_HW)