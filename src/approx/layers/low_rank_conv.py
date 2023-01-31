import torch
from torch import nn
from typing import Union

from .substituton import LAYER
from approx.utils.general import to_2tuple


class SeparableConv(nn.Module):
    """
    This is separable version of group convolution (in_channels=C, out_channels=M*C, groups=C)
    """

    def __init__(self, in_channels: int, num_bases: int,
                 kernel_size: tuple, stride: tuple, padding: tuple):
        super(SeparableConv, self).__init__()
        self.v_conv = nn.Conv2d(in_channels, in_channels * num_bases, (kernel_size[0], 1), (stride[0], 1),
                                (padding[0], 0),
                                bias=False, groups=in_channels)
        self.h_conv = nn.Conv2d(in_channels * num_bases, in_channels * num_bases, (1, kernel_size[1]), (1, stride[1]),
                                (0, padding[1]),
                                bias=False, groups=in_channels * num_bases)

    def forward(self, x):
        return self.h_conv(self.v_conv(x))


@LAYER.register_module()
class LowRankExpConvV1(nn.Module):
    """
    Scheme1 in https://arxiv.org/abs/1405.3866
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, tuple], stride: Union[int, tuple], padding: Union[int, tuple],
                 num_base: int, decomp: bool = False):
        super(LowRankExpConvV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.num_base = num_base
        self.s_conv = SeparableConv(self.in_channels, self.num_base, self.kernel_size, self.stride,
                                    self.padding) if decomp else nn.Conv2d(self.in_channels,
                                                                           self.in_channels * self.num_base,
                                                                           self.kernel_size, self.stride, self.padding,
                                                                           groups=self.in_channels)
        self.d_conv = nn.Conv2d(self.in_channels * self.num_base, self.out_channels, 1)

    def forward(self, x):
        return self.d_conv(self.s_conv(x))

    @property
    def bias(self):
        return self.d_conv.bias

    def decomp(self):
        if isinstance(self.s_conv, nn.Conv2d):
            s_conv = SeparableConv(self.in_channels, self.num_base, self.kernel_size, self.stride, self.padding)
            u, s, vh = torch.linalg.svd(self.s_conv.weight.data,
                                        full_matrices=False)  # (MC, 1, d, k), (MC, 1, k), (MC, 1, k, d)
            s = torch.sqrt(s)
            s_conv.v_conv.weight.data = (u[..., 0] * s[..., 0][..., None])[..., None]  # (MC, 1, d, 1)
            s_conv.h_conv.weight.data = (vh[..., 0, :] * s[..., 0][..., None])[..., None, :]  # (MC, 1, 1, d)
            self.s_conv = s_conv


@LAYER.register_module()
class LowRankExpConvV2(nn.Module):
    """
    Scheme2 in https://arxiv.org/abs/1405.3866
    """

    def __init__(self, in_channels: int, num_base: int,
                 kernel_size: tuple, stride: tuple, padding: tuple,
                 ):
        super(LowRankExpConvV2, self).__init__()
        self.v_conv = nn.Conv2d(in_channels, num_base, (kernel_size[0], 1), (stride[0], 1), (padding[0], 0), bias=False)
        self.h_conv = nn.Conv2d(num_base, num_base, (1, kernel_size[1]), (1, stride[1]), (0, padding[1]),
                                groups=num_base)

    def forward(self, x):
        return self.h_conv(self.v_conv(x))

    @property
    def bias(self):
        return self.h_conv.bias
