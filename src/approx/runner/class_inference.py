import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity

from .base import BaseRunner
from approx.utils.logger import get_logger
from approx.utils.serialize import load_model
from approx.utils.config import get_cfg
from approx.models import build_model
from approx.core import build_app
from approx.filters import build_filter

import os


class ClassInference(BaseRunner):
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.ori_model = build_model(cfg.model)
        self.app = build_app(cfg.app, deploy=True)
        self.filters = [build_filter(f_cfg) for f_cfg in cfg.filters]
        self.ckpt_path = os.path.join(cfg.work_dir, "opt.pth")

    def profile(self, model: nn.Module):
        x = torch.randn(16, 3, 224, 224).cuda()
        model = model.cuda()
        model.eval()
        model(x)  # warm up
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            model(x)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))

    def run(self):
        get_logger().info('Register...')
        self.model.register_switchable(self.app.src_type, self.filters)

        get_logger().info(
            f"There are {self.model.length_switchable} switchable submodules: {self.model._switchable_names}")

        get_logger().info('Initialize...')
        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

        # load_model(self.model, self.ckpt_path)

        if self.cfg.device == 'cuda':
            get_logger().info('Test Inference time')
            self.profile(self.ori_model)
            self.profile(self.model)
