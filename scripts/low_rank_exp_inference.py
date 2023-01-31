import torch
import torchvision
from torch import nn
from torch.profiler import profile, ProfilerActivity
from torchvision import transforms
from ptflops import get_model_complexity_info
import argparse
from datetime import datetime

from approx.utils.logger import get_logger, build_logger
from approx.utils.serialize import load_model
from approx.utils.config import get_cfg, init_cfg, update_cfg, save_cfg
from approx.models import build_model
from approx.core import build_app
from approx.filters import build_filter

import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Config file")
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint File")
    args = parser.parse_args()
    init_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = os.path.split(args.config)
    config_name = os.path.splitext(paths[-1])[0]
    work_dir = os.path.join("work_dir", config_name, datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(work_dir, exist_ok=True)
    log_name = f"class.log"
    log_file = os.path.join(work_dir, log_name)
    build_logger(log_file)
    update_cfg(work_dir=work_dir, device=device, log_file=log_file, ckpt_file=args.ckpt)


class ClassInference():
    def __init__(self):
        cfg = get_cfg()
        save_cfg(os.path.join(cfg.work_dir, "cfg.yaml"))
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.ori_model = build_model(cfg.model)
        self.app = build_app(cfg.app, deploy=True)
        self.filters = [build_filter(f_cfg) for f_cfg in cfg.filters]
        self.ckpt_path = cfg.ckpt_file

    def profile(self, model: nn.Module, desc: str):
        x = torch.randn(16, 3, 224, 224).cuda()
        model = model.cuda()
        model.eval()
        model(x)  # warm up
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            model(x)
        get_logger().info(
            f"{desc}:\n{prof.key_averages(group_by_input_shape=True).table(sort_by='cuda_time_total', row_limit=10)}")

    def classify(self, model: nn.Module, desc: str):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)
        device = self.cfg.device
        model = model.to(device)
        model.eval()
        total = 0
        correct = 0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc1 = 100 * correct / total
        get_logger().info(f"{desc}: Acc1={acc1}%")

    def run(self):
        get_logger().info('Register...')
        self.model.register_switchable(self.app.src_type, self.filters)

        get_logger().info(
            f"There are {self.model.length_switchable} switchable submodules: {self.model._switchable_names}")

        get_logger().info('Initialize...')
        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            self.model.set_switchable_module(idx, self.app.initialize, src=src)

        self.ori_model.init_weights()
        load_model(self.model, self.ckpt_path)

        # if self.cfg.device == 'cuda':
        #     self.profile(self.ori_model, 'Old Model')
        #     self.profile(self.model, 'New Model')
        #
        # macs, params = get_model_complexity_info(self.ori_model, (3, 224, 224))
        # get_logger().info(f'Old Model: macs={macs:<12}, params={params:<8}')
        # macs, params = get_model_complexity_info(self.model, (3, 224, 224))
        # get_logger().info(f'New Model: macs={macs:<12}, params={params:<8}')
        #
        # self.classify(self.ori_model, 'Oridinary Model')
        #
        # self.classify(self.model, 'New Model (Before PostProcess)')
        #
        for idx in range(self.model.length_switchable):
            src = self.model.get_switchable_module(idx)
            src.decomp()
        #
        # self.classify(self.model, 'New Model (After PostProcess)')
        self.profile(self.model, 'New Model (After PostProcess)')
        macs, params = get_model_complexity_info(self.model, (3, 224, 224))
        get_logger().info(f'New Model (After PostProcess): macs={macs:<12}, params={params:<8}')



def main():
    parse_args()
    runner = ClassInference()
    runner.run()


if __name__ == '__main__':
    main()
