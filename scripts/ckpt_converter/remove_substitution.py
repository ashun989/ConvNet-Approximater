import argparse
import os
from approx.models import build_model
from approx.core import build_app
from approx.filters import build_filter
from approx.utils import load_model, save_model
from approx.utils.config import Config

from approx.utils.logger import build_logger


def main():
    build_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='work_dir/tmp')

    args = parser.parse_args()
    cfg = Config()
    cfg.load_from_file(args.config)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    model = build_model(cfg.model)
    app = build_app(cfg.app, deploy=False)
    filters = [build_filter(f_cfg) for f_cfg in cfg.filters] if 'filters' in cfg else []

    model.register_switchable(app.src_type, filters)
    for idx in range(model.length_switchable):
        src = model.get_switchable_module(idx)
        model.set_switchable_module(idx, app.initialize, src=src)
    load_model(model, args.ckpt)
    for idx in range(model.length_switchable):
        sub = model.get_switchable_module(idx)
        model.set_switchable_module(idx, app.postprocess, sub=sub)
    save_model(model, os.path.join(args.out, 'no-sub.pth'))


if __name__ == '__main__':
    main()
