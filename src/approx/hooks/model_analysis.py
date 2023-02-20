from . import HOOK, Hook
from ptflops import get_model_complexity_info

from approx.utils.logger import get_logger


@HOOK.register_module()
class ModelAnalysis(Hook):
    def __init__(self, runner, priority,
                 input_shape=(3, 224, 224)):
        super(ModelAnalysis, self).__init__(runner, priority)
        self.input_shape = input_shape

    def after_run(self):
        macs, params = get_model_complexity_info(self.runner.model, self.input_shape)
        get_logger().info(f"Model Macs: {macs}, Params: {params}")
