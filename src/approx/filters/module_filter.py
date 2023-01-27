from abc import ABCMeta, abstractmethod
from torch import nn
from approx.utils.registry import build_from_cfg, Registry


class ModuleFilter(metaclass=ABCMeta):

    @abstractmethod
    def filter(self, module: nn.Module) -> bool:
        pass

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)


FILTER = Registry()


def build_filter(cfg: dict, **kwargs) -> ModuleFilter:
    return build_from_cfg(cfg, FILTER, **kwargs)
