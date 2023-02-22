from torch import nn

from approx.utils.registry import Registry, build_from_cfg


# class Parallel(nn.Module):
#     def __init__(self, *module_list: nn.Module):
#         super(Parallel, self).__init__()
#         self.net_list = nn.ModuleList(module_list)
#
#     def forward(self, x):
#         return [net(x) for net in self.net_list]


class Substitution(nn.Module):
    def __init__(self, old_module: nn.Module, new_module: nn.Module, use_old: bool = True):
        super(Substitution, self).__init__()
        self.old = old_module
        self.new = new_module
        # self.net = self.old if use_old else self.new
        self.use_old = use_old
        self.cache = {}

    @property
    def old_module(self) -> nn.Module:
        return self.old

    @property
    def new_module(self) -> nn.Module:
        return self.new

    def switch_new(self, remove_old=True):
        self.use_old = False
        if remove_old:
            delattr(self, "old")

    def switch_old(self, remove_new=False):
        self.use_old = True
        if remove_new:
            delattr(self, "new")

    def forward(self, x):
        if self.use_old:
            return self.old(x)
        return self.new(x)


LAYER = Registry()


def build_layer(cfg: dict, **kwargs) -> nn.Module:
    return build_from_cfg(cfg, LAYER, **kwargs)
