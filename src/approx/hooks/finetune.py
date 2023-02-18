import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import time
from collections import OrderedDict

from . import Hook, HOOK
from approx.models import build_model, SwitchableModel
from approx.layers import Substitution
from approx.utils import random_seed, distribute_bn, reduce_tensor, unwarp_model
from approx.utils.logger import get_logger

from timm.data import create_dataset, create_loader
from timm.optim import create_optimizer_v2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.utils.metrics import AverageMeter, accuracy

_default_dataset_args = dict(
    name='',
    root='path/to/dataset',
    batch_size=64
)

_default_data_config = dict(
    input_size=(3, 224, 224),
    interpolation='bicubic',
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    crop_pct=DEFAULT_CROP_PCT,
)

_default_optim_args = dict(
    opt='adamw',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.05,
    eps=1e-8
)

_default_other_args = dict(
    log_interval=50,
    num_epochs=20,
    num_workers=8,
    distributed=False,
    device='cuda:0',
    world_size=1,
    rank=0,
    local_rank=0,
    seed=42,
    sync_bn=False,
    dist_bn='reduce'
)


def cache_module_output(module: Substitution, input, output):
    module.cache['ori_output'] = output
    # get_logger().info(f'cache output: {output.shape}')


def get_l2_error(module: Substitution, input, output):
    B = output.shape[0]
    module.cache['norm'] = torch.norm((output - module.cache['ori_output']).reshape(B, -1), p=2, dim=1)


@HOOK.register_module()
class L2Reconstruct(Hook):
    def __init__(self,
                 runner, priority,
                 asym=True,
                 dataset_args={},
                 optim_args={},
                 data_config={},
                 other_args={}):
        super(L2Reconstruct, self).__init__(runner, priority)
        self.asym = asym
        self.model: SwitchableModel = self.runner.model
        self.dataset_args = _default_dataset_args
        self.dataset_args.update(dataset_args)
        self.optim_args = _default_optim_args
        self.optim_args.update(optim_args)
        self.data_config = _default_data_config
        self.data_config.update(data_config)
        self.other_args = _default_other_args
        self.other_args.update(other_args)
        if self.asym:
            self.ori_model = build_model(self.runner.cfg.model)

    def after_optimize(self):
        if self.other_args['distributed']:
            self.other_args['device'] = 'cuda:%d' % self.other_args['local_rank']
            torch.cuda.set_device(self.other_args['local_rank'])
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.other_args['world_size'] = torch.distributed.get_world_size()
            self.other_args['rank'] = torch.distributed.get_rank()
            get_logger().info(
                'Fine-tuning in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                % (self.other_args['rank'], self.other_args['world_size']))
        else:
            get_logger().info('Fine-tuning with a single process on 1 GPUs.')
        assert self.other_args['rank'] >= 0

        random_seed(self.other_args['seed'], self.other_args['rank'])

        if self.asym:
            self.ori_model.register_switchable(self.runner.app.src_type, self.runner.filters)
            self.ori_model.init_weights()
            for idx in range(self.ori_model.length_switchable):
                src = self.ori_model.get_switchable_module(idx)
                self.ori_model.set_switchable_module(idx, self.runner.app.initialize, src=src)
            for sub in self.ori_model.switchable_models():
                sub.switch_old(remove_new=True)
                sub.register_forward_hook(cache_module_output)
            for sub in self.model.switchable_models():
                sub.switch_new(remove_old=True)
                sub.register_forward_hook(get_l2_error)
            self.ori_model.eval()
            self.model.train()
        self.model.cuda()
        self.ori_model.cuda()

        if self.other_args['distributed'] and self.other_args['sync_bn']:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.other_args['local_rank'] == 0:
                get_logger().info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            self.model = DistributedDataParallel(self.model)

        dataset_train = create_dataset(split='train', **self.dataset_args)
        dataset_eval = create_dataset(split='validation', **self.dataset_args)
        loader_train = create_loader(dataset_train,
                                     input_size=self.data_config['input_size'],
                                     batch_size=self.dataset_args['batch_size'],
                                     is_training=True,
                                     no_aug=True,
                                     mean=self.data_config['mean'],
                                     std=self.data_config['std'],
                                     interpolation=self.data_config['interpolation'],
                                     num_workers=self.other_args['num_workers'],
                                     distributed=self.other_args['distributed'])
        loader_eval = create_loader(dataset_eval,
                                    input_size=self.data_config['input_size'],
                                    batch_size=self.dataset_args['batch_size'],
                                    is_training=False,
                                    no_aug=True,
                                    mean=self.data_config['mean'],
                                    std=self.data_config['std'],
                                    interpolation=self.data_config['interpolation'])

        optimizer = create_optimizer_v2(self.model, **self.optim_args)

        for epoch in range(self.other_args['num_epochs']):
            self.train_one_epoch(epoch, loader_train, optimizer)
            if self.other_args['distributed'] and self.other_args['dist_bn'] in ('broadcast', 'reduce'):
                if self.other_args['local_rank'] == 0:
                    get_logger().info("Distributing BatchNorm running means and vars")
                distribute_bn(self.model, self.other_args['world_size'], self.other_args['dist_bn'] == 'reduce')
            # TODO: remove forward hook after train_one_epoch
            # TODO: checkpoint
            self.validate(loader_eval)

    def train_one_epoch(self, epoch, loader_train, optimizer):
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()

        end = time.time()
        last_idx = len(loader_train) - 1
        num_updates = epoch * len(loader_train)
        for batch_idx, (input, target) in enumerate(loader_train):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            input, target = input.cuda(), target.cuda()
            if self.asym:
                with torch.no_grad():
                    self.ori_model(input)
                for ori_sub, sub in zip(self.ori_model.switchable_models(),
                                        unwarp_model(self.model).switchable_models()):
                    sub.cache['ori_output'] = ori_sub.cache['ori_output']
            else:
                for sub in unwarp_model(self.model).switchable_models():
                    sub.switch_old()
                    sub._forward_hooks = OrderedDict()
                    sub.register_forward_hook(cache_module_output)
                self.model.eval()
                with torch.no_grad():
                    self.model(input)
                for sub in unwarp_model(self.model).switchable_models():
                    sub.switch_new(remove_old=False)
                    sub._forward_hooks = OrderedDict()
                    sub.register_forward_hook(get_l2_error)
                self.model.train()
            self.model(input)
            total_norm = 0
            for sub in self.model.switchable_models():
                total_norm += sub.cache['norm']
            total_norm = torch.sum(total_norm)
            if not self.other_args['distributed']:
                losses_m.update(total_norm.item(), input.size(0))

            optimizer.zero_grad()
            total_norm.backward()
            optimizer.step()

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % self.other_args['log_interval'] == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if self.other_args['distributed']:
                    reduced_loss = reduce_tensor(total_norm.data, self.other_args['world_size'])
                    losses_m.update(reduced_loss.item(), input.size(0))

                if self.other_args['local_rank'] == 0:
                    get_logger().info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader_train),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * self.other_args['world_size'] / batch_time_m.val,
                            rate_avg=input.size(0) * self.other_args['world_size'] / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))
            end = time.time()
        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

    def validate(self, loader_eval):
        loss_fn = nn.CrossEntropyLoss().cuda()
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        self.model.eval()
        end = time.time()
        last_idx = len(loader_eval) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader_eval):
                last_batch = batch_idx == last_idx
                input, target = input.cuda(), target.cuda()
                output = self.model(input)
                loss = loss_fn(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                if self.other_args['distributed']:
                    reduced_loss = reduce_tensor(loss.data, self.other_args['world_size'])
                    acc1 = reduce_tensor(acc1, self.other_args['world_size'])
                    acc5 = reduce_tensor(acc5, self.other_args['world_size'])
                else:
                    reduced_loss = loss.data
                torch.cuda.synchronize()
                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if last_batch or batch_idx % self.other_args['log_interval'] == 0:
                    log_name = 'Test'
                    get_logger().info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
