import matplotlib.pyplot as plt
import torch

from approx.layers import LowRankExpConvV1
from torch import nn
import time


def inference(module, cfg):
    repeat = 100
    warmup = 10
    x = torch.randn(*cfg).cpu()
    module = module.cpu()
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        s_time = time.time()
        for _ in range(repeat):
            module(x)
        rtn = (time.time() - s_time) / repeat
    return rtn


def main():
    b = 128
    h = 224
    w = 224
    cfgs = [
        [(3, 64, 11, 4, 2), (b, 3, h, w)],
        [(64, 192, 5, 1, 2), (b, 64, h // 8, w // 8)],
        [(192, 384, 3, 1, 1), (b, 192, h // 16, w // 16)],
        [(384, 256, 3, 1, 1), (b, 384, h // 32, w // 32)],
        [(256, 256, 3, 1, 1), (b, 256, h // 32, w // 32)]
    ]
    for cfg_idx, cfg in enumerate(cfgs, 1):
        conv_cfg, x_cfg = cfg
        print(f"conv {conv_cfg}, input {x_cfg}")
        ord_conv = nn.Conv2d(*conv_cfg)
        std_time = inference(ord_conv, x_cfg)
        print(f"standard time: {std_time}")
        min_m = conv_cfg[0] // 2
        max_m = conv_cfg[0]
        sprs = []
        m_list = range(min_m, max_m, 10)
        for m in m_list:
            conv2 = LowRankExpConvV1(*conv_cfg, num_base=m, deploy=True)
            m_time = inference(conv2, x_cfg)
            print(f"num_bases: {m}, time: {m_time}")
            sprs.append(std_time / m_time)
        plt.plot(m_list, sprs)
        plt.xlabel('num_bases')
        plt.ylabel('speed-up ratios')
        plt.show()


if __name__ == '__main__':
    main()
