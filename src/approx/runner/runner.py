from approx.utils.config import get_cfg
from approx.models import build_model
from approx.core import build_app


class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.model = build_model(cfg.model)
        self.app = build_app(cfg.app)

    def run(self):
        self.model.register_switchable(self.app.src_type)
        for idx in self.model.length_switchable:
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

