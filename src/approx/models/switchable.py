from torch import nn
from typing import Callable

from approx.utils.registry import build_from_cfg, Registry


class SwitchableModel(nn.Module):
    def __init__(self):
        super(SwitchableModel, self).__init__()
        self._switchable_names: list[str] = []

    def register_switchable(self, src_type: type):
        cache = [("", self)]
        while cache:
            top = cache[0]
            del cache[0]
            if isinstance(top[1], src_type):
                self._switchable_names.append(top[0])
                continue  # Supposing src_type is not recursive
            for name, module in top[1].named_children():
                cache.append((f"{top[0]}.{name}", module))

    @property
    def length_switchable(self):
        return len(self._switchable_names)


    def set_switchable_module(self, index: int, func: Callable, **func_args):
        m_names = self._switchable_names[index].split('.')
        parent, curr = '.'.join(m_names[:-1]), m_names[-1]
        p_module = self.get_submodule(parent)
        if hasattr(p_module, curr):
            setattr(p_module, curr, func(**func_args))
        elif isinstance(p_module, nn.Sequential):
            p_module[int(curr)] = func(**func_args)
        else:
            assert False, f"module {p_module} does not have attr {curr}"

    def get_switchable_module(self, index: int):
        return self.get_submodule(self._switchable_names[index])


MODEL = Registry()


def build_model(cfg: dict) -> SwitchableModel:
    return build_from_cfg(cfg, MODEL)
