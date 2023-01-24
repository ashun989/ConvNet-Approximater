from abc import ABCMeta, abstractmethod
from torch import nn
from typing import Type

from approx.utils.registry import Registry, build_from_cfg
from approx.layers import build_layer, Substitution, LAYER


class Approximater(metaclass=ABCMeta):
    _src_type = ""
    _tgt_type = ""

    @property
    def tgt_type(self):
        return LAYER.get(self._tgt_type)

    @property
    def src_type(self):
        return LAYER.get(self._src_type)

    @abstractmethod
    def _get_tgt_args(self, src: nn.Module) -> dict:
        pass

    @abstractmethod
    def _fix_substitution(self, sub: Substitution):
        pass

    def initialize(self, src: nn.Module) -> Substitution:
        assert isinstance(src, self.src_type)
        tgt_args = self._get_tgt_args(src)
        cfg = dict(type=self.tgt_type)
        cfg.update(tgt_args)
        tgt = build_layer(cfg)
        sub = Substitution(src, tgt)
        self._fix_substitution(sub)
        return sub

    @abstractmethod
    def optimize(self, sub: Substitution):
        pass


APP = Registry()


def build_app(cfg: dict) -> Approximater:
    return build_from_cfg(cfg, APP)
