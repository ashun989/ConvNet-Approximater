import torch
import torch.nn.functional as F


def compare4d(y1: torch.Tensor, y2: torch.Tensor):
    assert y1.shape == y2.shape
    B, C, H, W = y1.shape
    y1 = y1.reshape(B, C, -1)
    y2 = y2.reshape(B, C, -1)
    a_err = torch.mean(torch.norm(y1 - y2, dim=(1, 2)))
    ref = torch.mean(torch.norm(y1, dim=(1, 2)))
    r_err = a_err / ref
    print(f"abs err: {a_err.item()}, rel arr: {r_err.item()}")
    assert r_err.item() < 1e-5


def test_equality1():
    B, C, H, W, N = 16, 256, 14, 14, 128
    d = 3
    x = torch.randn(B, C, H, W)
    w = torch.rand(N, C, d, d)
    y1 = F.conv2d(x, w) * 2
    y2 = F.conv2d(x, w * 2)
    compare4d(y1, y2)


def test_equality2():
    B, C, H, W, N, M = 16, 256, 14, 14, 128, 30
    d = 3
    x = torch.randn(B, C, H, W)
    v_w = torch.randn(M * C, 1, d, 1)
    h_w = torch.randn(M * C, 1, 1, d)
    s_w = v_w @ h_w
    y1 = F.conv2d(x, s_w, groups=C)
    tmp = F.conv2d(x, h_w, groups=C)
    y2 = F.conv2d(tmp, v_w, groups=M * C)
    compare4d(y1, y2)


def test_equality():
    N = 128
    C = 256
    M = 64
    d = 5
    pad = 2
    x = torch.randn(16, C, 14, 14)
    weights = torch.rand(N * C, M)
    bases = torch.rand(M, d ** 2)
    W = weights @ bases
    W = W.reshape(N, C, d, d)
    y1 = F.conv2d(x, W)

    s_w = bases.reshape(M, d, d).unsqueeze(0).expand(C, -1, -1, -1).reshape(C * M, 1, d, d)
    tmp = F.conv2d(x, s_w, groups=C)  # C*M, H', W'
    # d_w = torch.sum(weights.reshape(N, C, M), dim=1).unsqueeze(-1).unsqueeze(-1)  # (N, M, 1, 1)
    d_w = weights.reshape(N, C * M).unsqueeze(-1).unsqueeze(-1)  # (N, C*M, 1, 1)
    y2 = F.conv2d(tmp, d_w)
    compare4d(y1, y2)
