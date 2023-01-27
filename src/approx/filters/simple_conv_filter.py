from torch import nn

from .module_filter import ModuleFilter, FILTER


@FILTER.register_module()
class SimpleConvFilter(ModuleFilter):
    def __init__(self):
        pass

    def filter(self, module: nn.Module) -> bool:
        assert isinstance(module, nn.Conv2d)
        if module.groups > 1:
            return False
        if module.dilation[0] > 1 or module.dilation[1] > 1:
            return False
        if module.transposed:
            return False
        if module.bias is None:
            return False
        # TODO: add more cases
        return True
