import argparse
import torch
from torch import nn
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models.vision_transformer import _cfg
from timm.utils import accuracy, AverageMeter
from collections import OrderedDict
import time

from approx.models import build_model
from approx.utils.logger import get_logger, build_logger

mscan_t_cfg = dict(
    ori_ckpt_path="pretrained/mscan_t.pth",
    tgt_ckpt_path="pretrained/mscan_t_modified.pth",
    model_cfg=dict(
        type="MSCAN_Classifier",
        init_cfg="pretrained/mscan_t_modified.pth",
        num_channels=(32, 64, 160, 256),
        num_blocks=(3, 3, 5, 2),
        exp_ratios=(8, 8, 4, 4),
        drop_rate=0.0,
        drop_path_rate=0.1
    )
)

mscan_s_cfg = dict(
    ori_ckpt_path="pretrained/mscan_s.pth",
    tgt_ckpt_path="pretrained/mscan_s_modified.pth",
    model_cfg=dict(
        type="MSCAN_Classifier",
        init_cfg="pretrained/mscan_s_modified.pth",
        num_channels=(64, 128, 320, 512),
        num_blocks=(2, 2, 4, 2),
        exp_ratios=(8, 8, 4, 4),
        drop_rate=0.0,
        drop_path_rate=0.1
    )
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='t', help='t, s, l, b')
    parser.add_argument('--imagenet', type=str, default='/Zalick/Datasets/ILSVRC2012/', help='path to ImageNet')
    parser.add_argument('--batch-size', type=int, default=16)
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
                else:
                    b_parts = parts[4].split('_')
                    bidx = int(b_parts[0][-1])
                    sidx = int(b_parts[1][0])
                    new_prefix += f'attn.spatial_gating_unit.sd_convs.branches.{bidx}.conv{sidx}.'
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
    model.init_weights()

    param_count = sum([m.numel() for m in model.parameters()])
    get_logger().info('Model mscan_%s created, param count: %d' % (args.size, param_count))

    dataset = create_dataset(name='', root=args.imagenet, split='validation', is_training=False,
                             batch_size=args.batch_size)

    data_config = resolve_data_config(vars(args), default_cfg=_cfg(),
                                      use_test_size=True, verbose=True)

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'])

    model = model.cuda()
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    criterion = nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        model(input)
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % args.log_freq == 0:
                get_logger().info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=data_config['crop_pct'],
        interpolation=data_config['interpolation'])

    get_logger().info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


if __name__ == '__main__':
    build_logger()
    main()
