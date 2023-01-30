from torch import nn
import numpy as np
from typing import Tuple

from .module_filter import ModuleFilter, FILTER


@FILTER.register_module()
class IndicesFilter(ModuleFilter):
    def __init__(self, indices: Tuple[int, ...]):
        self.records = np.zeros(max(indices) + 1, np.bool_)
        self.records[list(indices)] = True
        self.curr = 1

    def filter(self, module: nn.Module) -> bool:
        passed = self.records[self.curr] if self.curr < len(self.records) else False
        self.curr += 1
        return passed
