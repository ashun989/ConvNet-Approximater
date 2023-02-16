from torch import nn

from . import LAYER


@LAYER.register_module()
class DummyLayer(nn.Module):
    pass
