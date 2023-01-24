from torch import nn

from .substituton import LAYER


@LAYER.register_module()
class SimpleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0):
        super(SimpleConv, self).__init__()
        self._in_c = in_channels
        self._out_c = out_channels
        self._k_size = kernel_size
        self._stride = stride
        self._padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

    @property
    def in_channels(self):
        return self._in_c

    @property
    def out_channels(self):
        return self._out_c

    @property
    def kernel_size(self):
        return self._k_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias
