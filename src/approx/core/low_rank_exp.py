import torch

from .approximater import Approximater, APP
from approx.layers import LowRankExpConvV1, LowRankExpConvV2, SimpleConv, Substitution
from approx.utils.logger import get_logger


@APP.register_module()
class LowRankExpV1(Approximater):
    """
    Scheme1 in [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
    """
    _src_type = "SimpleConv"
    _tgt_type = "LowRankExpConvV1"

    def __init__(self, max_iter, min_lmda, max_lmda, inc_rate=1.5):
        self.max_iter = max_iter
        self.lmda_list = torch.logspace(0, inc_rate, max_iter + 1)[1:] - 1
        self.lmda_list = self.lmda_list / self.lmda_list[-1] * (max_lmda - min_lmda) + min_lmda

    def _get_tgt_args(self, src: SimpleConv) -> dict:
        return dict(
            in_channels=src.in_channels,
            out_channels=src.out_channels,
            kernel_size=src.kernel_size,
            stride=src.stride,
            padding=src.padding,
            num_base=src.out_channels // 2
        )

    def _fix_substitution(self, sub: Substitution):
        src: SimpleConv = sub.old_module
        tgt: LowRankExpConvV1 = sub.new_module
        tgt.bias.data = src.bias.data

    def filter_construct_error(self, src: SimpleConv, tgt: LowRankExpConvV1, lmda: float):
        W = src.weight.data  # (N, C, d, d)
        C = src.weight.data.shape[1]
        alpha = tgt.d_conv.weight.data  # (N, M, 1, 1)
        alpha = alpha.reshape(*alpha.shape[:2])  # (N, M)
        bases = tgt.s_conv.weight.data  # (M, C, d, d)
        bases = bases.permute(2, 3, 0, 1)  # (d, d, M, C)
        _W = torch.matmul(alpha, bases).permute(2, 3, 0, 1)  # (N, C, d, d)
        err = torch.sum(torch.square(torch.linalg.norm(W - _W, dim=(2, 3), ord=2)))
        err2 = lmda * torch.sum(torch.linalg.norm(bases, dim=(0, 1), ord='nuc')) / C
        get_logger().debug(f"L2 error: {err}, nuclear norm: {err2}")
        return err + err2

    def optimize(self, sub: Substitution):
        epsilon = 1e-2
        last_err = self.filter_construct_error(sub.old_module, sub.new_module, self.lmda_list[0])
        get_logger().info(f"Total error: {last_err}")
        for iter, lmda in enumerate(self.lmda_list):
            # TODO: Alternative optimize
            err = self.filter_construct_error(sub.old_module, sub.new_module, lmda)
            get_logger().info(f"({iter}/{self.max_iter}), total error: {err}, lambda: {lmda}")
            if abs(last_err - err) < epsilon:
                break
            last_err = err
