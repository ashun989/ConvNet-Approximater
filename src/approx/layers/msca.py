from torch import nn
from .depth_seperable_conv import ParallelConv
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