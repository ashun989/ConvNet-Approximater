import time
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import apply_test_time_pool
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, set_jit_legacy

from approx.utils.logger import get_logger

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class ValidateHelper:
    def __init__(self, model, args):
        logger = get_logger()

        self.model = model
        self.args = args

        args.prefetcher = not args.no_prefetcher
        self.amp_autocast = suppress  # do nothing
        if args.amp:
            if has_native_amp:
                args.native_amp = True
            elif has_apex:
                args.apex_amp = True
            else:
                logger.warning("Neither APEX or Native Torch AMP is available.")
        assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
        if args.native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            logger.info('Validating in mixed precision with native PyTorch AMP.')
        elif args.apex_amp:
            logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            logger.info('Validating in float32. AMP not enabled.')

        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes

        self.data_config = resolve_data_config(args, model=model, use_test_size=True, verbose=True)

        dataset = create_dataset(
            root=args.data, name=args.dataset, split=args.split,
            load_bytes=args.tf_preprocessing, class_map=args.class_map)

        if args.valid_labels:
            with open(args.valid_labels, 'r') as f:
                self.valid_labels = {int(line.rstrip()) for line in f}
                self.valid_labels = [i in self.valid_labels for i in range(args.num_classes)]
        else:
            self.valid_labels = None

        if args.real_labels:
            self.real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
        else:
            self.real_labels = None

        test_time_pool = False
        if args.test_pool:
            _, test_time_pool = apply_test_time_pool(model, self.data_config, use_test_size=True)

        if test_time_pool:
            self.data_config['crop_pct'] = 1.0

        self.loader = create_loader(
            dataset,
            input_size=self.data_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=args.workers,
            crop_pct=self.data_config['crop_pct'],
            pin_memory=args.pin_mem,
            tf_preprocessing=args.tf_preprocessing)

    def validate(self):
        model = self.model
        args = self.args
        logger = get_logger()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        param_count = sum([m.numel() for m in model.parameters()])

        if args.legacy_jit:
            set_jit_legacy()

        model.cuda()
        if args.apex_amp:
            model = amp.initialize(model, opt_level='O1')

        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

        criterion = nn.CrossEntropyLoss().cuda()

        if args.test_pool:
            model, _ = apply_test_time_pool(model, self.data_config, use_test_size=True)

        if args.torchscript:
            torch.jit.optimized_execution(True)
            model = torch.jit.script(model)

        model.eval()
        with torch.no_grad():
            # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
            input = torch.randn((args.batch_size,) + tuple(self.data_config['input_size'])).cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            model(input)
            end = time.time()
            for batch_idx, (input, target) in enumerate(self.loader):
                if args.no_prefetcher:
                    target = target.cuda()
                    input = input.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                # compute output
                with self.amp_autocast():
                    output = model(input)

                if self.valid_labels is not None:
                    output = output[:, self.valid_labels]
                loss = criterion(output, target)

                if self.real_labels is not None:
                    self.real_labels.add_result(output)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                            batch_idx, len(self.loader), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses, top1=top1, top5=top5))

        if self.real_labels is not None:
            # real labels mode replaces topk values at the end
            top1a, top5a = self.real_labels.get_accuracy(k=1), self.real_labels.get_accuracy(k=5)
        else:
            top1a, top5a = top1.avg, top5.avg
        results = OrderedDict(
            top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
            param_count=round(param_count / 1e6, 2),
            img_size=self.data_config['input_size'][-1],
            crop_pct=self.data_config['crop_pct'],
            interpolation=self.data_config['interpolation'])

        logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
            results['top1'], results['top1_err'], results['top5'], results['top5_err']))

        return results
