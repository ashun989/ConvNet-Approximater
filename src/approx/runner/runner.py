from approx.utils.config import get_cfg, print_cfg
from approx.models import build_model
from approx.core import build_app
from approx.utils.logger import get_logger


class Runner:
    def __init__(self):
        cfg = get_cfg()
        print_cfg()
        self.model = build_model(cfg.model, device=cfg.device)
        self.app = build_app(cfg.app)

    def run(self):
        # Step1: Substitute old module with 2-branch module
        self.model.register_switchable(self.app.src_type)

        get_logger().info(
            f"There are {self.model.length_switchable} switchable submodules: {self.model._switchable_names}")

        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

        # Step2: Data-independent optimize
        for idx in range(self.model.length_switchable):
            self.app.optimize(self.model.get_switchable_module(idx))

        # Step3: Data-dependent optimize
        # TODO: Data-dependent optimize

        # Step4: Post process
        # TODO: post process
