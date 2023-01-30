import torch
from torch import nn
import numpy as np
import cvxpy as cp
import math

from .approximater import Approximater, APP
from approx.layers import LowRankExpConvV1, LowRankExpConvV2, Substitution
from approx.utils.logger import get_logger


@APP.register_module()
class LowRankExpV1(Approximater):
    """
    Scheme1 in [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
    """
    _src_type = nn.Conv2d
    _tgt_type = "LowRankExpConvV1"

    def __init__(self, num_bases, max_iter, lmda_length, min_lmda, max_lmda, inc_rate=1.5, deploy=False):
        super(LowRankExpV1, self).__init__(deploy=deploy)
        self.num_bases = num_bases
        self.curr = 0
        self.max_iter = max_iter
        assert max_lmda >= min_lmda >= 0.0
        self.lmda_list = np.logspace(0, inc_rate, lmda_length + 1)[1:] - 1
        self.lmda_list = self.lmda_list / self.lmda_list[-1] * (max_lmda - min_lmda) + min_lmda

    def _get_tgt_args(self, src: nn.Conv2d) -> dict:

        # Ordinary Conv: O(d^2CN)
        # Scheme1 Conv: O(MC(2d+N))

        # tmp1 = src.kernel_size[0] * src.kernel_size[1] * src.out_channels
        # tmp2 = src.kernel_size[0] + src.kernel_size[1] + src.out_channels
        # num_base = int(tmp1 / (self.speed_ratio * tmp2))
        # spr = tmp1 / (num_base * tmp2)
        # get_logger().info(f"theoretical speed-up ratio: {spr} (required: {self.speed_ratio}), num_base: {num_base}")
        num_base = self.num_bases[self.curr]
        self.curr += 1
        return dict(
            in_channels=src.in_channels,
            out_channels=src.out_channels,
            num_base=num_base,
            kernel_size=src.kernel_size,
            stride=src.stride,
            padding=src.padding
        )

    def _fix_substitution(self, sub: Substitution):
        src: nn.Conv2d = sub.old_module
        tgt: LowRankExpConvV1 = sub.new_module
        tgt.bias.data = src.bias.data

    def _get_bi_object(self,
                       filters: np.ndarray,  # (CN, d^2)
                       in_channels: int,
                       num_filters: int,
                       filter_size: int,
                       num_bases: int,
                       version: bool = True):
        if version:
            bases = cp.Variable((num_bases, filter_size ** 2))
            weights = cp.Parameter((in_channels * num_filters, num_bases))
        else:
            bases = cp.Parameter((num_bases, filter_size ** 2))
            weights = cp.Variable((in_channels * num_filters, num_bases))
        lmda = cp.Parameter(nonneg=True)
        nuc_list = []
        for m in range(num_bases):
            nuc_list.append(cp.normNuc(cp.reshape(bases[m], (filter_size, filter_size))))
        pred = weights @ bases  # (C*N, d^2)
        error = cp.sum(cp.norm2(filters - pred, axis=1))
        # error = cp.norm2(filters - pred)
        norm = lmda * sum(nuc_list)
        obj = cp.Minimize(error + norm)
        return cp.Problem(obj), dict(bases=bases, weights=weights, lmda=lmda, error=error, norm=norm, obj=obj)

    def optimize(self, sub: Substitution):
        logger = get_logger()
        epsilon = 1e-2
        src: nn.Conv2d = sub.old_module
        tgt: LowRankExpConvV1 = sub.new_module
        last_err = 0
        W = src.weight.data.numpy()  # (N, C, d, d)
        N, C, d = W.shape[:3]
        M = tgt.d_conv.in_channels
        W = W.transpose(1, 0, 2, 3)  # (C, N, d, d)
        W = W.reshape(C * N, d * d)
        problem1, cache1 = self._get_bi_object(W, C, N, d, M, True)
        problem2, cache2 = self._get_bi_object(W, C, N, d, M, False)
        assert problem1.is_dcp(), "problem1 is not DCP!"
        assert problem2.is_dcp(), "problem2 is not DCP!"
        # too many paramters to use DPP
        # if not problem1.is_dcp(dpp=True):
        #     logger.warn("problem1 is not DPP!")
        # if not problem2.is_dcp(dpp=True):
        #     logger.warn("problem2 is not DPP!")
        # cache1['bases'].value = np.random.rand(M, d**2)
        cache1['weights'].value = np.ones((N * C, M)) / M
        # cache2['weights'].value = np.ones((N*C, M)) / M
        # TODO: choose a better solver
        logger.info(f"lambda list: {self.lmda_list}")
        for e, lmda in enumerate(self.lmda_list):
            cache1['lmda'].value = lmda
            cache2['lmda'].value = lmda
            for iter in range(1, self.max_iter + 1):
                # Fix weights, update bases
                problem1.solve(ignore_dpp=True)
                total_err = cache1['obj'].value
                logger.info(f"[lamda: {lmda}]({iter}/{self.max_iter})[1], total error: {total_err}")
                cache2['bases'].value = cache1['bases'].value
                # Fix bases, update weights
                problem2.solve(ignore_dpp=True)
                total_err = cache2['obj'].value
                logger.info(f"[lamda: {lmda}]({iter}/{self.max_iter})[2], total error: {total_err}")
                cache1['weights'].value = cache2['weights'].value
                if abs(last_err - total_err) < epsilon:
                    break
                last_err = total_err
            svdvals = torch.linalg.svdvals(torch.from_numpy(cache1['bases'].value).reshape(M, d, d))
            svdvals = svdvals**2
            energy = torch.mean(svdvals[:, 0] / torch.sum(svdvals, dim=1))
            logger.info(f"energy = {energy.item()}")
        tmp = torch.from_numpy(cache1['bases'].value)  # (M, d^2)
        tmp = tmp.reshape(M, d, d)
        u, s, vh = torch.linalg.svd(tmp, full_matrices=False)  # (M, d, k), (M, k), (M, k, d)
        s = torch.sqrt(s)
        u = (u[:, :, 0] * s[:, 0][:, None])[:, :, None]  # (M, d, 1)
        vh = (vh[:, 0, :] * s[:, 0][:, None])[:, None, :]  # (M, 1, d)
        tgt.s_conv.v_conv.weight.data = u.unsqueeze(1).expand(-1, C, -1, -1)  # (M, C, d, 1)
        tgt.s_conv.h_conv.weight.data = vh.unsqueeze(1)  # (M, 1, 1, d)
        tmp = torch.from_numpy(cache2['weights'].value)  # (C*N, M)
        tmp = torch.sum(tmp.reshape(C, N, M), dim=0)
        tgt.d_conv.weight.data[:, :, 0, 0] = tmp


    def _postprocess(self, sub: Substitution):
        pass
