import torch
from torch import nn
from typing import Optional
import warnings

from .general import check_file
from .config import get_cfg
from .logger import get_logger


def load_model(model: nn.Module, ckpt_file: str, device: Optional[str] = None, verbose: bool = True) -> bool:
    if check_file(ckpt_file):
        if device is None:
            device = get_cfg().device
        ckpt = torch.load(ckpt_file, map_location=device)
        if 'state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if verbose:
            get_logger().info(f"Load state dict from {ckpt_file}")
        if missing:
            warnings.warn(f"Missing Keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")
        return True
    return False


def save_model(model: nn.Module, ckpt_file: str, verbose: bool = True):

    torch.save(dict(state_dict=model.state_dict()), ckpt_file)
    if verbose:
        get_logger().info(f"model state dict is saved in {ckpt_file}")
