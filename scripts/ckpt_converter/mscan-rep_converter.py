import argparse
import torch

from approx.models import build_model
from approx.core import build_app
from approx.utils.logger import get_logger, build_logger
from approx.utils import load_model
from approx.classification.validate import ValidateHelper
from approx.hooks.class_eval_hook import _default_eval_cfg
from approx.utils.config import Config

mscan_t_cfg = dict(
    ori_ckpt_path="pretrained/mscan_t_d1_ft.pth",
    tgt_ckpt_path="pretrained/mscan_t_d1_ft_modified.pth",
    model_cfg=dict(
        type="MSCAN_Classifier",
        num_channels=(32, 64, 160, 256),
        num_blocks=(3, 3, 5, 2),
        exp_ratios=(8, 8, 4, 4),
        drop_rate=0.0,
        drop_path_rate=0.1
    )
)

mscan_s_cfg = dict(
    ori_ckpt_path="pretrained/mscan_s_d1_ft.pth",
    tgt_ckpt_path="pretrained/mscan_s_d1_ft_modified.pth",
    model_cfg=dict(
        type="MSCAN_Classifier",
        num_channels=(64, 128, 320, 512),
        num_blocks=(2, 2, 4, 2),
        exp_ratios=(8, 8, 4, 4),
        drop_rate=0.0,
        drop_path_rate=0.1
    )
)

app_cfg = dict(
    type="MscaRep",
    decomp=1,
    fix=False
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='t', help='t, s, l, b')
    parser.add_argument('--imagenet', type=str, default='/Zalick/Datasets/ILSVRC2012/', help='path to ImageNet')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num-gpu', type=int, default=2)
    parser.add_argument('--log-freq', default=300, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')

    args = parser.parse_args()

    assert args.size in ('t', 's')
    mscan_cfg = eval(f'mscan_{args.size}_cfg')

    ori_ckpt = torch.load(mscan_cfg["ori_ckpt_path"])
    tgt_ckpt = {}

    for k, v in ori_ckpt['state_dict'].items():
        k: str
        if k.startswith('patch_embed'):
            parts = k.split('.')
            idx = int(parts[0][-1])
            new_prefix = f'backbone.layers.{idx - 1}.0.'
            same_part = '.'.join(parts[1:])
            tgt_ckpt[new_prefix + same_part] = v
        elif k.startswith('block'):
            parts = k.split('.')
            idx = int(parts[0][-1])
            block_idx = int(parts[1])
            new_prefix = f'backbone.layers.{idx - 1}.1.{block_idx}.'
            if len(parts) >= 5 and parts[3] == 'spatial_gating_unit' and parts[4] != 'conv0':
                if parts[4] == 'conv3':
                    new_prefix += 'attn.spatial_gating_unit.channel_mix.'
                    same_part = '.'.join(parts[5:])
                    tgt_ckpt[new_prefix + same_part] = v
                elif parts[4] == 'rep_conv':
                    new_prefix += f'attn.spatial_gating_unit.sd_convs.'
                    same_part = '.'.join(parts[5:])
                    tgt_ckpt[new_prefix + same_part] = v
            elif parts[2] == 'mlp' and parts[3] == 'dwconv':
                new_prefix += 'mlp.dconv.'
                same_part = '.'.join(parts[5:])
                tgt_ckpt[new_prefix + same_part] = v
            else:
                same_part = '.'.join(parts[2:])
                tgt_ckpt[new_prefix + same_part] = v
        elif k.startswith('norm'):
            parts = k.split('.')
            idx = int(parts[0][-1])
            new_prefix = f'backbone.layers.{idx - 1}.2.'
            same_part = '.'.join(parts[1:])
            tgt_ckpt[new_prefix + same_part] = v
        else:
            tgt_ckpt[k] = v

    torch.save(tgt_ckpt, mscan_cfg["tgt_ckpt_path"])

    model = build_model(mscan_cfg["model_cfg"])
    app = build_app(app_cfg, deploy=True)
    model.register_switchable(app.src_type, filters=[])
    for idx, m in enumerate(model.switchable_modules()):
        model.set_switchable_module(idx, app.initialize, src=m)
    load_model(model, mscan_cfg["tgt_ckpt_path"])
    param_count = sum([m.numel() for m in model.parameters()])
    get_logger().info('Model mscan_%s created, param count: %d' % (args.size, param_count))

    eval_cfg = Config()
    eval_cfg.update(_default_eval_cfg)
    eval_cfg.update(dict(batch_size=args.batch_size,
                         data=args.imagenet,
                         workers=args.workers,
                         num_gpu=args.num_gpu,
                         log_freq=args.log_freq,
                         crop_pct=0.9,
                         mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)))

    eval_helper = ValidateHelper(model, eval_cfg)
    eval_helper.validate()


if __name__ == '__main__':
    build_logger()
    main()
