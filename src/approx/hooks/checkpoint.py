from . import Hook, HOOK
from approx.utils.config import Config
from approx.utils import check_file, save_model, load_model, parse_path

import os


@HOOK.register_module()
class CkptHook(Hook):
    def __init__(self, runner, priority,
                 ckpt_cfg):
        super(CkptHook, self).__init__(runner, priority)
        self.ckpt_cfg = Config()
        for stage in self.stages:
            if stage in ckpt_cfg:
                cur = ckpt_cfg[stage]
                assert cur['action'] in ('save', 'load')
                if cur['action'] == 'load':
                    assert check_file(cur['path'])
                else:
                    dir, _, _ = parse_path(cur['path'])
                    assert os.path.isdir(dir)
                self.ckpt_cfg[stage] = cur
            else:
                self.ckpt_cfg[stage] = None

    def save_or_load(self, cfg):
        if cfg is not None:
            if cfg["action"] == "save":
                save_model(self.runner.model, cfg["path"])
            else:
                load_model(self.runner.model, cfg["path"])

    def before_run(self):
        self.save_or_load(self.ckpt_cfg.before_run)

    def after_register(self):
        self.save_or_load(self.ckpt_cfg.after_register)

    def after_initialize(self):
        self.save_or_load(self.ckpt_cfg.after_initialize)

    def after_optimize(self):
        self.save_or_load(self.ckpt_cfg.after_optimize)

    def after_run(self):
        self.save_or_load(self.ckpt_cfg.after_run)
