import argparse
import os
from datetime import datetime

import torch.cuda

from approx.runner import Runner
from approx.utils.config import init_cfg, update_cfg, get_cfg
from approx.utils.logger import build_logger, get_logger
from approx.utils import check_file, random_seed, parse_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approaches to decompose the convolutional layers for better inference time")
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. If it is used,"
                             "the optimization step will be skipped, `app.initialize` arg `deploy` will be True,"
                             "and this checkpoint will be load after initialization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    deploy = False
    if args.checkpoint:
        deploy = True
        assert check_file(args.checkpoint)

    init_cfg(args.config)

    distributed = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    world_size = 1
    rank = 0
    local_rank = 0

    work_dir = None
    log_file = None
    config_name = None

    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
        local_rank = int(os.environ['LOCAL_RANK'])

    if local_rank == 0:
        config_name = parse_filename(args.config)
        work_dir = os.path.join("work_dir", config_name, datetime.now().strftime('%Y%m%d%H%M%S'))
        os.makedirs(work_dir, exist_ok=True)
        log_name = "eval.log" if deploy else "train.log"
        log_file = os.path.join(work_dir, log_name)
        build_logger(log_file)
        print(log_file)

    if distributed:
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        get_logger().info(
            'Running in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (rank, world_size))
    else:
        get_logger().info('Running with a single process on 1 GPUs.')

    assert rank >= 0

    random_seed(args.seed, rank)

    update_cfg(work_dir=work_dir, device=device, log_file=log_file,
               checkpoint=args.checkpoint, deploy=deploy, config_name=config_name,
               distributed=distributed, world_size=world_size, rank=rank, local_rank=local_rank,
               seed=args.seed)


def main():
    parse_args()
    cfg = get_cfg()
    runner = Runner(deploy=cfg.deploy)
    runner.run()


if __name__ == '__main__':
    main()
