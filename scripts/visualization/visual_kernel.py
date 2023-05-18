import argparse
import os.path

from approx.models import build_model
from approx.core import build_app
from approx.layers import CascadeConv
from approx.utils import load_model
from approx.utils.logger import build_logger
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm

mscan_t_cfg = dict(
    type="MSCAN_Classifier",
    init_cfg="pretrained/mscan_t_modified.pth",
    num_channels=(32, 64, 160, 256),
    num_blocks=(3, 3, 5, 2),
    exp_ratios=(8, 8, 4, 4),
    drop_rate=0.0,
    drop_path_rate=0.1
)

app_cfg_0 = dict(
    type="MscaRep",
    decomp=0,
    fix=True
)

app_cfg_1 = dict(
    type="MscaRep",
    decomp=1,
    fix=False
)


def plot_square(weight: np.ndarray):
    dim = weight.shape[0]
    H, W = weight.shape[2:]
    white = np.ones((H, W), dtype=np.uint8)
    # plt.imshow(white, cmap='gray')
    # plt.show()
    width = int(np.sqrt(dim))
    height = int(np.ceil(dim / width))
    fig, axs = plt.subplots(height, width, figsize=(width, height), subplot_kw=dict(xticks=[], yticks=[]),
                            gridspec_kw=dict(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.95))
    r, c = 0, 0
    for d in tqdm(range(dim)):
        conv_map = weight[d, 0]
        conv_map[H // 2, W // 2] = np.min(conv_map)
        conv_map = (conv_map - np.min(conv_map)) / (np.max(conv_map) - np.min(conv_map))
        r = d // width
        c = d - (r * width)
        # print(f"d = {d}, (r, c) = ({r}, {c})")
        axs[r, c].imshow(conv_map, interpolation='nearest')
        # axs[r, c].axis('off')
    c += 1
    while c < width:
        axs[r, c].imshow(white, cmap='binary')
        axs[r, c].axis('off')
        c += 1
    # plt.subplots_adjust(wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95)
    # plt.tight_layout()


def plot_cascade_conv(cascade_conv: CascadeConv, idx, output_dir):
    # fig, ax = plt.subplots()
    # dim = cascade_conv.conv1.weight.data.shape[0]
    h_conv = cascade_conv.conv1.weight.data.numpy()
    v_conv = cascade_conv.conv2.weight.data.numpy()
    conv = v_conv @ h_conv
    # for d in tqdm(range(dim)):
    #     conv_map = conv[d, 0]
    #     conv_map = (conv_map - np.min(conv_map)) / (np.max(conv_map) - np.min(conv_map))
    #     plt.imshow(conv_map)
    #     plt.savefig(os.path.join(output_dir, f"MSSA_i{idx}_d{d}.png"))
    plot_square(conv)
    plt.savefig(os.path.join(output_dir, f"MSSA_i{idx}.pdf"))


def plot_depth_wise_conv(conv: nn.Conv2d, idx, output_dir):
    dim = conv.weight.data.shape[0]
    # conv_n = conv.weight.data.numpy()
    # for d in tqdm(range(dim)):
    #     conv_map = conv_n[d, 0]
    #     conv_map = (conv_map - np.min(conv_map)) / (np.max(conv_map) - np.min(conv_map))
    #     plt.imshow(conv_map)
    #     plt.savefig(os.path.join(output_dir, f"MSSA_i{idx}_d{d}.png"))
    plot_square(conv.weight.data.numpy())
    plt.savefig(os.path.join(output_dir, f"MSSA_i{idx}.pdf"))


def plot_1(idx, output_dir):
    model = build_model(mscan_t_cfg)
    app = build_app(app_cfg_0, deploy=True)
    model.register_switchable(app.src_type, filters=[])
    for i in range(model.length_switchable):
        src = model.get_switchable_module(i)
        model.set_switchable_module(i, app.initialize, src=src)
    load_model(model, "pretrained/mscan_t_d0_fix.pth")
    src = model.get_switchable_module(idx)
    plot_depth_wise_conv(src.sd_convs[0], idx, output_dir)


def plot_2(idx, output_dir):
    model = build_model(mscan_t_cfg)
    app = build_app(app_cfg_1, deploy=True)
    model.register_switchable(app.src_type, filters=[])
    for i in range(model.length_switchable):
        src = model.get_switchable_module(i)
        model.set_switchable_module(i, app.initialize, src=src)
    load_model(model, "pretrained/mscan_t_d1.pth")
    src = model.get_switchable_module(idx)
    plot_cascade_conv(src.sd_convs, idx, output_dir)


def plot_3(idx, output_dir):
    model = build_model(mscan_t_cfg)
    app = build_app(app_cfg_1, deploy=True)
    model.register_switchable(app.src_type, filters=[])
    for i in range(model.length_switchable):
        src = model.get_switchable_module(i)
        model.set_switchable_module(i, app.initialize, src=src)
    load_model(model, "pretrained/mscan_t_d1_ft_modified.pth")
    src = model.get_switchable_module(idx)
    plot_cascade_conv(src.sd_convs, idx, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int)

    args = parser.parse_args()

    output_dir1 = os.path.join("work_dir", "kernels", f"idx{args.idx}", "ori")
    output_dir2 = os.path.join("work_dir", "kernels", f"idx{args.idx}", "d1")
    output_dir3 = os.path.join("work_dir", "kernels", f"idx{args.idx}", "d1_ft")
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(output_dir3, exist_ok=True)

    plot_1(args.idx, output_dir1)
    plot_2(args.idx, output_dir2)
    plot_3(args.idx, output_dir3)


if __name__ == '__main__':
    build_logger()
    main()
