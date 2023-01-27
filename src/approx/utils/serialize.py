import torch
from torch import nn
from typing import Optional

from .logger import get_logger
from .general import check_file


def load_model(model: nn.Module, ckpt_file: str, device: Optional[str] = None) -> bool:
    if check_file(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        if 'state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        logger = get_logger()
        logger.info(f"Load state dict from {ckpt_file}")
        if missing:
            logger.warn(f"Missing Keys: {missing}")
        if unexpected:
            logger.warn(f"Unexpected keys: {unexpected}")
        return True
    return False


def save_model(model: nn.Module, ckpt_file: str):
    torch.save(dict(state_dict=model.state_dict()), ckpt_file)
    get_logger().info(f"model state dict is saved in {ckpt_file}")
