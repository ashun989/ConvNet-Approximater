from abc import ABCMeta, abstractmethod
from torch import nn

from approx.utils.registry import Registry, build_from_cfg
from approx.layers import build_layer, Substitution


class Approximater(metaclass=ABCMeta):
    _src_type = nn.Module
    _tgt_type = nn.Module

    @property
    def tgt_type(self) -> type:
        return self._tgt_type

    @property
    def src_type(self) -> type:
        return self._src_type

    @abstractmethod
    def _get_tgt_args(self, src: nn.Module) -> dict:
        pass

    def initialize(self, src: nn.Module) -> Substitution:
        assert isinstance(src, self.src_type)
        tgt_args = self._get_tgt_args(src)
        cfg = dict(type=self.tgt_type)
        cfg.update(tgt_args)
        tgt = build_layer(cfg)
        return Substitution(src, tgt)


APP = Registry()


def build_app(cfg: dict) -> Approximater:
    return build_from_cfg(cfg, APP)
