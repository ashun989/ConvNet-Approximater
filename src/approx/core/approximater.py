from abc import ABCMeta, abstractmethod
from torch import nn
from typing import Union

from approx.utils.registry import Registry, build_from_cfg
from approx.layers import build_layer, Substitution, LAYER


class Approximater(metaclass=ABCMeta):
    _src_type = ""
    _tgt_type = ""

    def __init__(self, deploy=False):
        self.deploy = deploy

    @property
    def tgt_type(self):
        if isinstance(self._tgt_type, type):
            return self._tgt_type
        return LAYER.get(self._tgt_type)

    @property
    def src_type(self):
        if isinstance(self._src_type, type):
            return self._src_type
        return LAYER.get(self._src_type)

    @abstractmethod
    def _get_tgt_args(self, src: nn.Module) -> dict:
        pass

    @abstractmethod
    def _fix_substitution(self, sub: Substitution):
        pass

    def initialize(self, src: nn.Module) -> Union[Substitution, nn.Module]:
        assert isinstance(src, self.src_type)
        tgt_args = self._get_tgt_args(src)
        cfg = dict(type=self.tgt_type)
        cfg.update(tgt_args)
        tgt = build_layer(cfg)
        sub = Substitution(src, tgt)
        self._fix_substitution(sub)
        if self.deploy:
            return sub.new_module
        return sub

    @abstractmethod
    def optimize(self, sub: Substitution):
        pass

    @abstractmethod
    def _postprocess(self, sub: Substitution):
        pass

    def postprocess(self, sub: Substitution) -> nn.Module:
        self._postprocess(sub)
        return sub.new_module


APP = Registry()


def build_app(cfg: dict, **kwargs) -> Approximater:
    return build_from_cfg(cfg, APP, **kwargs)
