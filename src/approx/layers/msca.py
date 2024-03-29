from torch import nn
import torch.autograd.profiler as profiler

from .depth_seperable_conv import ParallelConv, CascadeConv
from .substituton import LAYER


@LAYER.register_module()
class MSCA(nn.Module):
    def __init__(self,
                 num_channel,
                 k1_size,
                 k_sizes):
        super(MSCA, self).__init__()
        self.num_channel = num_channel
        self.k1_size = k1_size
        self.k_sizes = k_sizes
        self.conv0 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel,
                               kernel_size=k1_size, padding=k1_size // 2, groups=num_channel)

        paddings = [k // 2 for k in k_sizes]
        self.sd_convs = ParallelConv(num_channel, k_sizes, paddings, len(k_sizes), True, True)
        self.channel_mix = nn.Conv2d(num_channel, num_channel, 1)

    def forward(self, x):
        return x * self.channel_mix(self.sd_convs(self.conv0(x)))

    def switchable_layer(self):
        return self.sd_convs


@LAYER.register_module()
class MSCAProfile(MSCA):
    def forward(self, x):
        attn = x.clone()
        with profiler.record_function("CONV0"):
            attn = self.conv0(attn)
        with profiler.record_function("SD_CONVS"):
            attn = self.sd_convs(attn)
        with profiler.record_function("CHANNEL_MIX"):
            attn = self.channel_mix(attn)
        return attn * x


# @LAYER.register_module()
# class MSCAFinal(nn.Module):
#     def __init__(self,
#                  num_channel,
#                  k1_size,
#                  k_sizes):
#         super(MSCAFinal, self).__init__()
#         self.num_channel = num_channel
#         self.k1_size = k1_size
#         self.k_sizes = k_sizes
#         self.conv0 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel,
#                                kernel_size=k1_size, padding=k1_size // 2, groups=num_channel)
#
#         paddings = [k // 2 for k in k_sizes]
#         self.sd_convs = CascadeConv(num_channel, max(k_sizes), paddings, True, False)
#         self.channel_mix = nn.Conv2d(num_channel, num_channel, 1)
#
#     def forward(self, x):
#         return x * self.channel_mix(self.sd_convs(self.conv0(x)))
