import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from approx.layers import Substitution
from approx.layers import MSCA, FixPaddingBias, CascadeConv, ParallelConv
from approx.utils.general import to_2tuple

from .approximater import Approximater, APP


@APP.register_module()
class MscaRep(Approximater):
    """
    Reparameter MSCA module.
    """
    _src_type = "MSCA"
    _tgt_type = "MSCA"

    def __init__(self, decomp: int, fix: bool, deploy: bool = False):
        super().__init__(deploy=deploy)
        assert 0 <= decomp <= 4
        self.decomp = decomp
        self.fix = fix

    def _get_tgt_args(self, src: MSCA) -> dict:
        return dict(
            num_channel=src.num_channel,
            k1_size=src.k1_size,
            k_sizes=src.k_sizes
        )

    def _fix_substitution(self, sub: Substitution):
        src: MSCA = sub.old_module
        tgt: MSCA = sub.new_module
        tgt.conv0.weight.data = src.conv0.weight.data
        tgt.conv0.bias.data = src.conv0.bias.data
        tgt.channel_mix.weight.data = src.channel_mix.weight.data
        tgt.channel_mix.bias.data = src.channel_mix.bias.data
        max_k_size = max(src.k_sizes)
        padding = max_k_size // 2
        if self.decomp == 0:
            sd_conv = nn.Conv2d(src.num_channel, src.num_channel, max_k_size,
                                padding=padding, groups=src.num_channel)
        elif self.decomp == 1:
            sd_conv = CascadeConv(src.num_channel, max_k_size, padding, True, False)
        else:
            sd_conv = ParallelConv(src.num_channel, max_k_size, padding,
                                   self.decomp, False, False)
        if self.fix:
            fix = FixPaddingBias(src.num_channel, padding)
            tgt.sd_convs = nn.Sequential(sd_conv, fix)
        else:
            tgt.sd_convs = sd_conv

    @staticmethod
    def _sum_bias(wx_2, bx_1, bx_2, pad_2=None):
        """
        :param wx_2: (C, 1, H2, 1), second kernel weight.
        :param bx_1: (C), first kernel bias.
        :param bx_2: (C), second kernel bias.
        :param pad_2: p, top-bottom padding of wx_2. default: p = H2 // 2.
        :return center_bias:
        :return border_bias_res: res[0] and res[1] denotes top border and bottom border respectively, of which each contains `pad_2` line residues.
        """
        # tmp = torch.sum(wx_2, dim=(-2, -1))
        # tmp = tmp * bx_1.view(1, -1)  # bx.view(1, -1) for groups=0, and bx.view(-1, 1) for groups=C
        # tmp = torch.sum(tmp, -1)
        # return tmp + bx_2
        assert wx_2.shape[1] == 1
        assert wx_2.shape[-1] == 1
        h2 = wx_2.shape[-2]
        C = wx_2.shape[0]
        if pad_2 is None:
            pad_2 = h2 // 2
        center_bias = torch.squeeze(torch.sum(wx_2, dim=(-2, -1)), dim=-1) * bx_1 + bx_2  # C
        border_bias_res = torch.zeros((2, C, pad_2), device=center_bias.device)
        for i in range(1, pad_2 + 1):
            border_bias_res[0, :, (pad_2 - i)] = -torch.squeeze(torch.sum(wx_2[:, :, :i, :], dim=(-2, -1)),
                                                                dim=-1) * bx_1
            border_bias_res[1, :, i - 1] = -torch.squeeze(torch.sum(wx_2[:, :, -i:, :], dim=(-2, -1)), dim=-1) * bx_1
        return center_bias, border_bias_res

    @staticmethod
    def _merge_res(*res_list):
        max_p = max([res.shape[-1] for res in res_list])
        c = res_list[0].shape[1]
        device = res_list[0].device
        merged = torch.zeros((2, c, max_p), device=device)
        for res in res_list:
            p = res.shape[-1]
            merged[0, :, :p] += res[0, :, :]
            merged[1, :, -p:] += res[1, :, :]
        return merged

    @staticmethod
    def _mul_weight(wx_1, wx_2):
        """
        :param wx_1: (C, 1, 1, W)
        :param wx_2: (C, 1, H, 1)
        """
        assert wx_1.shape[1] == wx_2.shape[1] == 1
        return wx_2 @ wx_1

    @staticmethod
    def _pad2d_zeros(kernel, shape):
        shape = to_2tuple(shape)
        w = max(shape[-1] - kernel.shape[-1], 0) // 2
        h = max(shape[-2] - kernel.shape[-2], 0) // 2
        p2d = (w, w, h, h)
        return F.pad(kernel, p2d, "constant", 0)

    @staticmethod
    def get_equivalent_kernel(module: ParallelConv):
        w1_lst = []
        b1_lst = []
        w2_lst = []
        b2_lst = []
        for b in module.branches:
            if hasattr(b, "conv1"):
                w1_lst.append(b.conv1.weight)
                b1_lst.append(b.conv1.bias)
                w2_lst.append(b.conv2.weight)
                b2_lst.append(b.conv2.bias)

        hw = max(w1_lst[-1].shape[2:])
        k_id = np.zeros((module.dim, 1, hw, hw), dtype=np.float32)
        k_id[:, 0, hw // 2, hw // 2] = 1
        k_id = torch.from_numpy(k_id).to(w1_lst[0].device)

        weight = k_id
        b_lst = []
        r_lst = []
        for w1, b1, w2, b2 in zip(w1_lst, b1_lst, w2_lst, b2_lst):
            weight += MscaRep._pad2d_zeros(MscaRep._mul_weight(w1, w2), hw)
            b, r = MscaRep._sum_bias(w2, b1, b2)
            b_lst.append(b)
            r_lst.append(r)

        bias = sum(b_lst)
        res = MscaRep._merge_res(*r_lst)
        return weight, bias, res

    def optimize(self, sub: Substitution):
        src: MSCA = sub.old_module
        tgt: MSCA = sub.new_module
        weight, bias, res = MscaRep.get_equivalent_kernel(src.sd_convs)
        if self.decomp == 0:
            sd_conv: nn.Conv2d = tgt.sd_convs[0] if self.fix else tgt.sd_convs
            sd_conv.weight.data = weight
            sd_conv.bias.data = bias
        else:
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)  #
            s = torch.sqrt(s)
            if self.decomp == 1:
                sd_conv: CascadeConv = tgt.sd_convs[0] if self.fix else tgt.sd_convs
                sd_conv.conv1.weight.data = (vh[..., 0, :] * s[..., 0][..., None])[..., None, :]
                sd_conv.conv2.weight.data = (u[..., 0] * s[..., 0][..., None])[..., None]
                sd_conv.conv2.bias.data = bias
            else:
                sd_conv: ParallelConv = tgt.sd_convs[0] if self.fix else tgt.sd_convs
                for j in range(self.decomp):
                    sd_conv.branches[j].conv1.weight.data = (vh[..., j, :] * s[..., j][..., None])[..., None, :]
                    sd_conv.branches[j].conv2.weight.data = (u[..., j] * s[..., j][..., None])[..., None]
                sd_conv.branches[-1].conv2.bias.data = bias
        if self.fix:
            fix: FixPaddingBias = tgt.sd_convs[1]
            fix.res.data = res

    def _postprocess(self, sub: Substitution):
        pass
