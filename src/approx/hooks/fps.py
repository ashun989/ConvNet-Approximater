from . import Hook, HOOK
from .finetune import _default_dataset_args, _default_data_config, combine_config

import time
import torch
import numpy as np

from timm.data import create_dataset, create_loader


@HOOK.register_module()
class Fps(Hook):
    def __init__(self, runner, priority,
                 repeat_times=1,
                 log_interval=50,
                 num_workers=8,
                 use_torch_benchmark=False,
                 dataset_args={},
                 data_config={}):
        super(Fps, self).__init__(runner, priority)
        self.repeat_times = repeat_times
        self.log_interval = log_interval
        self.num_workers = num_workers
        self.use_torch_benchmark = use_torch_benchmark
        self.dataset_args = combine_config(_default_dataset_args, dataset_args)
        self.data_config = combine_config(_default_data_config, data_config)

    def after_run(self):
        repeat_times = self.repeat_times
        # set cudnn_benchmark
        torch.backends.cudnn.benchmark = self.use_torch_benchmark
        benchmark_dict = dict()
        overall_fps_list = []
        for time_index in range(repeat_times):
            print(f'Run {time_index + 1}:')
            # build the dataloader
            dataset = create_dataset(split='validation', **self.dataset_args)
            data_loader = create_loader(
                dataset,
                input_size=self.data_config.input_size,
                batch_size=self.dataset_args.batch_size,
                is_training=False,
                no_aug=True,
                mean=self.data_config.mean,
                std=self.data_config.std,
                interpolation=self.data_config.interpolation,
                num_workers=self.num_workers,
                distributed=False)

            # build the model and load checkpoint
            model = self.runner.model
            model.cuda()
            model.eval()
            # the first several iterations may be very slow so skip them
            num_warmup = 5
            pure_inf_time = 0
            total_iters = 200

            num_imgs = 0
            # benchmark with 200 image and take the average
            for i, (input, _) in enumerate(data_loader):
                input = input.cuda()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    model(input)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                if i >= num_warmup:
                    pure_inf_time += elapsed
                    num_imgs += input.size(0)
                    if (i + 1) % self.log_interval == 0:
                        # fps = (i + 1 - num_warmup) / pure_inf_time
                        fps = num_imgs / pure_inf_time
                        print(f'Done iter [{i + 1:<3}/ {total_iters}], '
                              f'fps: {fps:.2f} img / s')

                if (i + 1) == total_iters:
                    # fps = (i + 1 - num_warmup) / pure_inf_time
                    fps = num_imgs / pure_inf_time
                    print(f'Overall fps: {fps:.2f} img / s\n')
                    benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                    overall_fps_list.append(fps)
                    break
        benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
        benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
        print(f'Average fps of {repeat_times} evaluations: '
              f'{benchmark_dict["average_fps"]}')
        print(f'The variance of {repeat_times} evaluations: '
              f'{benchmark_dict["fps_variance"]}')
