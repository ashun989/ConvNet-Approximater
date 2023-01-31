import os.path

from approx.utils.config import get_cfg, print_cfg, save_cfg
from approx.models import build_model
from approx.core import build_app
from approx.filters import build_filter
from approx.utils.logger import get_logger
from approx.utils.serialize import save_model, load_model
from .base import BaseRunner


class Runner(BaseRunner):
    def __init__(self):
        cfg = get_cfg()
        print_cfg()
        save_cfg(os.path.join(cfg.work_dir, "cfg.yaml"))
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.app = build_app(cfg.app)
        self.filters = [build_filter(f_cfg) for f_cfg in cfg.filters]
        self.ckpt_path = os.path.join(cfg.work_dir, "opt.pth")

    def run(self):
        # Step1: Substitute old module with 2-branch module
        get_logger().info('Register...')
        self.model.register_switchable(self.app.src_type, self.filters)

        get_logger().info(
            f"There are {self.model.length_switchable} switchable submodules: {self.model._switchable_names}")

        get_logger().info('Initialize...')
        self.model.init_weights()
        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

        # Step2: Data-independent optimize
        get_logger().info('Optimize...')
        for idx in range(self.model.length_switchable):
            self.app.optimize(self.model.get_switchable_module(idx))

        save_model(self.model, os.path.join(self.cfg.work_dir, "opt-b.pth"))

        # Step3: Data-dependent optimize
        # TODO: Data-dependent optimize

        # Step4: Post process
        get_logger().info('PostProcess...')
        for idx in range(self.model.length_switchable):
            sub = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.postprocess, sub=sub)

        save_model(self.model, self.ckpt_path)
