import os
import random
import argparse
import math

import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, ToTensor, Normalize

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

from PIL import Image

from timm.data.transforms_factory import transforms_imagenet_eval

from approx.models import build_model
from approx.layers import MSCA
from approx.core import build_app
from approx.utils import load_model, random_seed, check_file, parse_path
from approx.utils.logger import build_logger

from typing import List, Callable, Tuple

from imagenet_dict import ImageNetDict

imagenet_dict = ImageNetDict('scripts/visualization/imagenet.txt')


class AttnAndGrads:

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_attn))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_grad))

    def save_attn(self, module: MSCA, input, output):
        with torch.no_grad():
            attn = module.channel_mix(module.sd_convs(module.conv0(input[0])))
        if self.reshape_transform is not None:
            attn = self.reshape_transform(attn)
        self.activations.append(attn.cpu().detach())

    def save_grad(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class MscaAttnCAM(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None) -> None:
        super(MscaAttnCAM, self).__init__(model, target_layers, use_cuda, reshape_transform)
        self.activations_and_grads.release()
        self.activations_and_grads = AttnAndGrads(
            self.model, target_layers, reshape_transform
        )

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:
        spatial_weighted_activations = np.maximum(grads, 0) * activations
        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        print(f"target_categories: {target_categories}")
        if targets is None:
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer), target_categories[0]


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='mscacam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'mscacam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--img', type=str, default='')
    # parser.add_argument('--convert', type=str, default='rep')
    parser.add_argument('--idx', type=int, default=12)

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


methods = \
    {"gradcam": GradCAM,
     "hirescam": HiResCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM,
     "layercam": LayerCAM,
     "mscacam": MscaAttnCAM,
     "fullgrad": FullGrad,
     "gradcamelementwise": GradCAMElementWise}


def get_model(args):
    model = build_model(mscan_t_cfg)
    if args.convert == 'rep':
        app = build_app(app_cfg_0, deploy=True)
    else:
        app = build_app(app_cfg_1, deploy=True)
    model.register_switchable(app.src_type, filters=[])

    assert -1 <= args.idx < model.length_switchable

    for i in range(model.length_switchable):
        src = model.get_switchable_module(i)
        model.set_switchable_module(i, app.initialize, src=src)
    if args.convert == 'rep':
        load_model(model, "pretrained/mscan_t_d0_fix.pth")
    elif args.convert == 'd1':
        load_model(model, "pretrained/mscan_t_d1.pth")
    else:
        load_model(model, "pretrained/mscan_t_d1_ft_modified.pth")
    return model


def gen_cam_loop(model, cam_algorithm, input_tensor, targets, rgb_img, input_dir, out_dir, args):
    input_cls = targets[0].category
    if args.idx == -1:
        target_layers = [model.get_switchable_module(i) for i in range(model.length_switchable)]
    else:
        target_layers = [model.get_switchable_module(args.idx)]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam, pred_cls = cam(input_tensor=input_tensor,
                                      targets=targets,
                                      aug_smooth=args.aug_smooth,
                                      eigen_smooth=args.eigen_smooth)
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        rgb_img = np.asarray(rgb_img)
        rgb_img = np.float32(rgb_img) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    input_name = parse_path(input_dir)[1]
    cv2.imwrite(
        os.path.join(out_dir, f'{input_name}_{args.method}_{args.idx}_{args.convert}_{pred_cls}_{input_cls}_cam.jpg'),
        cam_image)


def main():
    args = get_args()

    # assert args.convert in ('rep', 'd1', 'ft')

    if args.seed is not None:
        random_seed(args.seed, 0)

    out_dir = 'work_dir/cams'
    os.makedirs(out_dir, exist_ok=True)

    if args.img:
        check_file(args.img)
        input_dir = args.img
    else:
        root_dir = 'data/ILSVRC2012/val'
        classes_dir = os.listdir(root_dir)
        input_dir = os.path.join(root_dir, classes_dir[random.randint(0, len(classes_dir) - 1)])
        pictures = os.listdir(input_dir)
        input_dir = os.path.join(input_dir, pictures[random.randint(0, len(pictures) - 1)])
    print(input_dir)

    input_cls = imagenet_dict.dir2cls[os.path.split(parse_path(input_dir)[0])[-1]]
    print(f"input_cls: {input_cls}")

    input_image = Image.open(input_dir).convert('RGB')
    scale_size = int(math.floor(224 / 0.9))
    transform1 = Compose([
        Resize(scale_size, InterpolationMode.BICUBIC),
        CenterCrop(224)])
    transform2 = Compose([ToTensor(),
                          Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    rgb_img = transform1(input_image)
    input_tensor = transform2(rgb_img).unsqueeze(0)
    targets = [ClassifierOutputTarget(input_cls)]

    cam_algorithm = methods[args.method]

    for i in range(3):
        if i == 0:
            args.convert = 'rep'
        elif i == 1:
            args.convert = 'd1'
        else:
            args.convert = 'ft'
        model = get_model(args)
        gen_cam_loop(model, cam_algorithm, input_tensor, targets, rgb_img, input_dir, out_dir, args)

    cv2.imwrite(
        os.path.join(out_dir, f'{parse_path(input_dir)[1]}.jpg'), np.asarray(rgb_img)[:, :, ::-1])


if __name__ == '__main__':
    build_logger()
    main()
