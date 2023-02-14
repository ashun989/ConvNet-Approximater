from torch import nn


class CascadeConv(nn.Module):
    """
    Cascade 1xd and dx1 depth-wise convolution.
    """

    def __init__(self, dim, kernel_size, padding, bias, first_bias):
        super(CascadeConv, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=first_bias)
        self.conv2 = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ParallelConv(nn.Module):
    """
    Multi-branch of `CascadeConv`
    """

    def __init__(self, dim, kernel_sizes, paddings, nbranch, all_bias, identity):
        super(ParallelConv, self).__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * nbranch
        if isinstance(paddings, int):
            paddings = [paddings] * nbranch
        assert len(kernel_sizes) == nbranch
        assert len(paddings) == nbranch
        if all_bias:
            self.branches = nn.ModuleList(
                [CascadeConv(dim=dim, kernel_size=kernel_sizes[i], padding=paddings[i], bias=True, first_bias=True)
                 for i in range(nbranch)])
        else:
            self.branches = nn.ModuleList(
                [CascadeConv(dim=dim, kernel_size=kernel_sizes[i], padding=paddings[i],
                             bias=True if i == nbranch - 1 else False, first_bias=False)
                 for i in range(nbranch)])
        if identity:
            self.branches.append(nn.Identity())

    def forward(self, x):
        tmp = [b(x) for b in self.branches]
        return sum(tmp)
