from .priority import get_priority
from approx.utils.registry import Registry, build_from_cfg


class Hook():
    def __init__(self, runner, priority):
        self.runner = runner
        self._priority = get_priority(priority)

    stages = (
        'before_run', 'after_register',
        'after_initialize', 'after_optimize',
        'after_run'
    )

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def priority(self):
        return self._priority

    def before_run(self):
        pass

    def after_register(self):
        pass

    def after_initialize(self):
        pass

    def after_optimize(self):
        pass

    def after_run(self):
        pass


HOOK = Registry()


def build_hook(cfg: dict, **kwargs) -> Hook:
    return build_from_cfg(cfg, HOOK, **kwargs)
