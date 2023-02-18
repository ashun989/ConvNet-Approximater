from torch import nn

from approx.utils.registry import Registry, build_from_cfg


class Parallel(nn.Module):
    def __init__(self, *module_list: nn.Module):
        super(Parallel, self).__init__()
        self.net_list = nn.ModuleList(module_list)

    def forward(self, x):
        return [net(x) for net in self.net_list]


class Substitution(nn.Module):
    def __init__(self, old_module: nn.Module, new_module: nn.Module):
        super(Substitution, self).__init__()
        self.old = old_module
        self.new = new_module
        self.net = self.old
        self.cache = {}

    @property
    def old_module(self) -> nn.Module:
        return self.old

    @property
    def new_module(self) -> nn.Module:
        return self.new

    def switch_new(self, remove_old=True):
        self.net = self.new
        if remove_old:
            delattr(self, "old")

    def switch_old(self, remove_new=False):
        self.net = self.old
        if remove_new:
            delattr(self, "new")

    def forward(self, x):
        return self.net(x)


LAYER = Registry()


def build_layer(cfg: dict, **kwargs) -> nn.Module:
    return build_from_cfg(cfg, LAYER, **kwargs)
