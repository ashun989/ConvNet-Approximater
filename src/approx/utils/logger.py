import logging
from typing import Optional
from .config import get_cfg


class DummyLogger:
    def trace(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass


_dummy_logger = DummyLogger()


def get_logger():
    cfg = get_cfg()
    if not cfg.distributed or cfg.local_rank == 0:
        return logging.getLogger()
    return _dummy_logger


def build_logger(log_file: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
