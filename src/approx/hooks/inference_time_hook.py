import torch
from torch.profiler import profile, ProfilerActivity

from . import Hook, HOOK
from approx.utils.logger import get_logger


@HOOK.register_module()
class InferenceTimeHook(Hook):
    def __init__(self, runner, priority, infer_cfg):
        super(InferenceTimeHook, self).__init__(runner, priority)
        self.input_size = infer_cfg.pop("input_size", (256, 3, 224, 224))
        self.profile_args = infer_cfg.pop("profile_args",
                                          dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]))
        self.key_args = infer_cfg.pop("key_args", {})
        # if 'record_shapes' in self.profile_args and self.profile_args['record_shapes']:
        #     self.key_args['group_by_input_shape'] = True
        # else:
        #     self.key_args['group_by_input_shape'] = False
        self.table_args = infer_cfg.pop("table_args", dict(sort_by='cuda_time_total', row_limit=10))
        get_logger().info(f"InferenceTimeHook Config:\n"
                          f"input_size: {self.input_size}\n"
                          f"profile_args: {self.profile_args}\n"
                          f"key_args: {self.key_args}\n"
                          f"table_args: {self.table_args}")

    def after_run(self):
        model = self.runner.model.cuda()
        model = model.eval()
        x = torch.randn(*self.input_size).cuda()
        model(x)
        with profile(**self.profile_args) as prof:
            model(x)
        get_logger().info(
            f"Profile Result:\n{prof.key_averages(**self.key_args).table(**self.table_args)}")
