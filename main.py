import argparse
import os.path
from datetime import datetime


import torch.cuda

from approx.utils import init_cfg, update_cfg, get_cfg
from approx.runner import Runner, ClassInference
from approx.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approaches to decompose the convolutional layers for better inference time")
    parser.add_argument("--config", required=True, type=str, help="Config file")
    parser.add_argument("--mode", type=str, default='train', help="train,class")
    args = parser.parse_args()

    assert args.mode in ['train', 'class']

    init_cfg(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paths = os.path.split(args.config)
    config_name = os.path.splitext(paths[-1])[0]
    work_dir = os.path.join("work_dir", config_name)
    os.makedirs(work_dir, exist_ok=True)
    log_name = f"{args.mode}-{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    log_file = os.path.join(work_dir, log_name)
    build_logger(log_file)
    update_cfg(work_dir=work_dir, device=device, log_file=log_file, mode=args.mode)


def main():
    parse_args()
    cfg = get_cfg()
    if cfg.mode == 'train':
        runner = Runner()
    else:
        runner = ClassInference()
    runner.run()


if __name__ == '__main__':
    main()
