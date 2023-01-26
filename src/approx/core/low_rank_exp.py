import torch
import numpy as np
import cvxpy as cp

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
        err2 = lmda * torch.sum(torch.linalg.norm(bases[..., 0], dim=(0, 1), ord='nuc'))
        get_logger().debug(f"L2 error: {err}, nuclear norm: {err2}")
        return err + err2

    def _get_bi_object(self,
                       filters: np.ndarray,
                       num_filters: int,
                       num_bases: int,
                       k_size: int,
                       version: bool = True):
        if version:
            bases = [cp.Variable((k_size, k_size)) for _ in range(num_filters)]
            weights = cp.Parameter((num_filters, num_bases))
        else:
            bases = [cp.Parameter((k_size, k_size)) for _ in range(num_filters)]
            weights = cp.Variable((num_filters, num_bases))
        lmda = cp.Parameter(nonneg=True)
        l2_list = []
        nuc_list = []
        for n in range(num_filters):
            m_adds = []
            for m in range(num_bases):
                m_adds.append(weights[n, m] * cp.vec(bases[m]))
            l2_list.append(cp.norm2(cp.vec(filters[n]) - sum(m_adds)))
        for m in range(num_bases):
            nuc_list.append(cp.normNuc(bases[m]))
        error = sum(l2_list)
        norm = lmda * sum(nuc_list)
        obj = cp.Minimize(error + norm)
        return cp.Problem(obj), dict(bases=bases, weights=weights, lmda=lmda, error=error, norm=norm)

    def optimize(self, sub: Substitution):
        logger = get_logger()
        epsilon = 1e-2
        src: SimpleConv = sub.old_module
        tgt: LowRankExpConvV1 = sub.new_module
        last_err = 0
        W = src.weight.data.numpy()  # (N, C, d, d)
        N, C, d = W.shape[:3]
        M = tgt.d_conv.in_channels
        W = W.reshape(-1, d, d)
        problem1, cache1 = self._get_bi_object(W, N * C, M, d, True)
        problem2, cache2 = self._get_bi_object(W, N * C, M, d, False)
        assert problem1.is_dcp(), "problem1 is not DCP!"
        assert problem2.is_dcp(), "problem2 is not DCP!"
        if not problem1.is_dcp(ddp=True):
            logger.warn("problem1 is not DDP!")
        if not problem2.is_dcp(ddp=True):
            logger.warn("problem2 is not DDP!")
        cache1['weights'].value = np.ones((N * C, M)) / M
        logger.info("Begin optimizing")
        for iter, lmda in enumerate(self.lmda_list):
            # Fix weights, update bases
            cache1['lmda'].value = lmda
            problem1.solve()
            total_err = cache1['error'].value + cache1['norm'].value
            logger.info(f"({iter}/{self.max_iter})[1], lambda: {lmda}, L2 error: {cache1['error']}, "
                        f"regular term: {cache1['norm']}, total error: {total_err}")
            for n in range(N * C):
                cache2['bases'][n].value = cache1['bases'][n].value
            # Fix bases, update weights
            cache2['lmda'].value = lmda
            problem2.solve()
            total_err = cache2['error'].value + cache2['norm'].value
            logger.info(f"({iter}/{self.max_iter})[2], lambda: {lmda}, L2 error: {cache1['error']}, "
                        f"regular term: {cache1['norm']}, total error: {total_err}")
            cache1['weights'].value = cache2['weights']

            if abs(last_err - total_err) < epsilon:
                break
            last_err = total_err
        logger.info("Finish optimizing")
