from . import Hook, HOOK
from approx.layers import LowRankExpConvV1


@HOOK.register_module()
class LowRankExpV1Decomp(Hook):
    def __init__(self, runner, priority):
        super(LowRankExpV1Decomp, self).__init__(runner, priority)

    def after_run(self):
        for s_module in self.runner.model.switchable_modules():
            assert isinstance(s_module, LowRankExpConvV1)
            s_module.decomp()
   