from torch import nn
from typing import Dict

from . import APP, Approximater
from ..layers import Substitution


@APP.register_module()
class Dummy(Approximater):
    _src_type = "DummyLayer"
    _tgt_type = "DummyLayer"

    def __init__(self, deploy):
        super(Dummy, self).__init__(deploy)

    def _get_tgt_args(self, src: nn.Module) -> Dict:
        return {}

    def _fix_substitution(self, sub: Substitution):
        pass

    def optimize(self, sub: Substitution):
        pass

    def _postprocess(self, sub: Substitution):
        pass
