import argparse
import os.path

import torch.cuda

from approx.utils import init_cfg, update_cfg
from approx.runner import Runner
from approx.utils.logger import build_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approaches to decompose the convolutional layers for better inference time")
    parser.add_argument("--config", required=True, type=str, help="Config file")
    args = parser.parse_args()
    init_cfg(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paths = os.path.split(args.config)
    work_dir = os.path.join("work_dir", *paths[1:-1])
    config_name = os.path.splitext(paths[-1])[0]
    update_cfg(work_dir=work_dir, device=device)
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, config_name + '.log')
    build_logger(log_file)


def main():
    parse_args()
    runner = Runner()
    runner.run()


if __name__ == '__main__':
    main()
