import time
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import apply_test_time_pool
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, set_jit_legacy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

from . import Hook, HOOK
from approx.utils.logger import get_logger
from approx.classification import ValidateHelper

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

_default_eval_cfg = dict(
    batch_size=128,
    workers=4,
    num_gpu=2,
    log_freq=50,

    input_size=(3, 224, 224),
    num_classes=1000,
    pool_size=None,
    crop_pct=DEFAULT_CROP_PCT,
    interpolation='bicubic',
    fixed_input_size=True,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,

    data='path/to/dataset',
    dataset='',  # dataset type (default: ImageFolder/ImageTar if empty)
    split='validation',
    dataset_download=False,

    class_map='',  # path to class to idx mapping file (default: "")
    test_pool=False,  # enable test time pool
    no_prefetcher=False,
    pin_mem=False,
    channels_last=False,
    amp=False,
    apex_amp=False,
    native_amp=False,
    tf_preprocessing=False,
    legacy_jit=False,
    real_labels='',  # Real labels JSON file for imagenet evaluation
    valid_labels='',  # Valid label indices txt file for validation of partial label space
)


@HOOK.register_module()
class ClassEvalHook(Hook):
    def __init__(self, runner, priority, eval_cfg):
        super(ClassEvalHook, self).__init__(runner, priority)
        self.eval_cfg = eval_cfg.copy()
        self.parse_cfg()
        self.helper = ValidateHelper(self.runner.model, self.eval_cfg)

    def parse_cfg(self):
        for k, v in _default_eval_cfg.items():
            if k not in self.eval_cfg:
                self.eval_cfg[k] = v

    def after_run(self):
        self.helper.validate()
