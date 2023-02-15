import argparse
import os.path
from datetime import datetime

import torch.cuda

from approx.runner import Runner
from approx.utils.config import init_cfg, update_cfg, get_cfg
from approx.utils.logger import build_logger
from approx.utils import check_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approaches to decompose the convolutional layers for better inference time")
    parser.add_argument("--config", required=True, type=str, help="Config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file, "
                             "the optimization step will be skipped if used.")

    args = parser.parse_args()

    deploy = False
    if args.checkpoint:
        deploy = True
        assert check_file(args.checkpoint)

    init_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = os.path.split(args.config)
    config_name = os.path.splitext(paths[-1])[0]
    work_dir = os.path.join("work_dir", config_name, datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(work_dir, exist_ok=True)
    log_name = "eval.log" if deploy else "train.log"
    log_file = os.path.join(work_dir, log_name)
    build_logger(log_file)
    update_cfg(work_dir=work_dir, device=device, log_file=log_file, checkpoint=args.checkpoint, deploy=deploy)


def main():
    parse_args()
    cfg = get_cfg()
    runner = Runner(deploy=cfg.deploy)
    runner.run()


if __name__ == '__main__':
    main()
