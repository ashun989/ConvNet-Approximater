import os.path
from typing import List

from approx.models import build_model
from approx.core import build_app
from approx.filters import build_filter
from approx.hooks import build_hook, Hook
from approx.utils.logger import get_logger
from approx.utils.serialize import save_model, load_model
from approx.utils.general import is_method_overridden
from approx.utils.config import get_cfg, print_cfg, save_cfg
from .base import BaseRunner


class Runner(BaseRunner):
    def __init__(self, deploy: bool = False):
        cfg = get_cfg()
        print_cfg()
        save_cfg(os.path.join(cfg.work_dir, "cfg.yaml"))
        self.deploy = deploy
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.app = build_app(cfg.app, deploy=deploy)
        self.filters = [build_filter(f_cfg) for f_cfg in cfg.filters] if 'filters' in cfg else []
        self.hooks: List[Hook] = []
        self.output_path = os.path.join(cfg.work_dir, cfg.config_name + ".pth")

        if hasattr(cfg, "hooks"):
            for h_cfg in cfg.hooks:
                self.register_hook(h_cfg)
            get_logger().info(self.hook_info())

    def run(self):
        self.call_hook("before_run")

        get_logger().info('Register...')
        self.model.register_switchable(self.app.src_type, self.filters)

        get_logger().info(
            f"There are {self.model.length_switchable} switchable submodules: {self.model._switchable_names}")

        self.call_hook("after_register")

        get_logger().info('Initialize...')
        self.model.init_weights()
        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

        self.call_hook("after_initialize")

        if self.deploy:
            load_model(self.model, self.cfg.checkpoint)
        else:
            get_logger().info('Optimize...')
            for sub in self.model.switchable_models():
                self.app.optimize(sub)

            self.call_hook("after_optimize")

            # Step3: Post process
            get_logger().info('PostProcess...')
            for idx in range(self.model.length_switchable):
                sub = self.model.get_switchable_module(idx)
                self.model.set_switchable_module(idx, self.app.postprocess, sub=sub)
            save_model(self.model, self.output_path)

        self.call_hook("after_run")

    def register_hook(self, hook_cfg):
        hook = build_hook(hook_cfg, runner=self)
        idx = 0
        ok = False
        for h in self.hooks:
            if hook.priority < h.priority:
                ok = True
                break
            idx += 1
        if ok:
            self.hooks.insert(idx, hook)
        else:
            self.hooks.append(hook)

    def call_hook(self, hook_stage):
        for h in self.hooks:
            getattr(h, hook_stage)()

    def hook_info(self):
        info = {}
        for stage in Hook.stages:
            info[stage] = []
            for h in self.hooks:
                if is_method_overridden(stage, Hook, h):
                    info[stage].append((h.name, h.priority))

        info_str = "\n"
        for k, v in info.items():
            info_str += f"Stage {k}:\n"
            info_str += f"{'Name':^20}|{'Prio':^10}\n"
            info_str += '-' * 30 + '\n'
            for pair in v:
                info_str += f"{pair[0]:^20}|{pair[1]:^10}\n"
            info_str += '-' * 30 + '\n'
        return info_str
