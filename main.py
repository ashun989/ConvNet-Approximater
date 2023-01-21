import argparse
from approx.utils import init_cfg, update_cfg
from approx.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approaches to decompose the convolutional layers for better inference time")
    parser.add_argument("--config", required=True, type=str, help="Config file")
    args = parser.parse_args()
    init_cfg(args.config)


def main():
    parse_args()
    runner = Runner()
    runner.run()


if __name__ == '__main__':
    main()
