import torch
from torch import nn
from typing import Callable, List, Union, Iterable

from approx.utils.registry import build_from_cfg, Registry
from approx.utils.logger import get_logger
from approx.utils.serialize import load_model
from approx.filters import ModuleFilter
from approx.layers import Substitution


class SwitchableModel(nn.Module):
    def __init__(self, init_cfg):
        super(SwitchableModel, self).__init__()
        self._switchable_names: List[str] = []
        self.init_cfg = init_cfg

    def init_weights(self):
        if isinstance(self.init_cfg, str):
            load_model(self, self.init_cfg)

    def register_switchable(self, src_type: type, filters: List[ModuleFilter], verbose: bool = False):
        cache = [(name, module) for name, module in self.named_children()]
        while cache:
            top = cache[0]
            del cache[0]
            if isinstance(top[1], src_type):
                passed = True
                for f in filters:
                    if not f(top[1]):
                        passed = False
                        if verbose:
                            get_logger().info(f"{top[0]} is filtered out by {f.__class__.__name__}")
                        break
                if passed:
                    self._switchable_names.append(top[0])
                continue  # Supposing src_type is not recursive
            for name, module in top[1].named_children():
                cache.append((f"{top[0]}.{name}", module))

    @property
    def length_switchable(self) -> int:
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

    def get_switchable_module(self, index: int) -> Union[nn.Module, Substitution]:
        return self.get_submodule(self._switchable_names[index])

    def switchable_models(self) -> Iterable[Union[nn.Module, Substitution]]:
        for idx in range(self.length_switchable):
            yield self.get_switchable_module(idx)

    def freeze_except(self, index: int):
        for p in self.parameters():
            p.requires_grad = False

        except_name = self._switchable_names[index]
        cache = [(name, module) for name, module in self.named_children()]
        while cache:
            top = cache[0]
            del cache[0]
            if top[0] == except_name:
                if hasattr(top[1], "switchable_layer"):
                    for p in top[1].switchable_layer().parameters():
                        p.requires_grad = True
                else:
                    for p in top[1].parameters():
                        p.requires_grad = True
                break
            for name, module in top[1].named_children():
                cache.append((f"{top[0]}.{name}", module))

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


MODEL = Registry()


def build_model(cfg: dict) -> SwitchableModel:
    model = build_from_cfg(cfg, MODEL)
    return model
