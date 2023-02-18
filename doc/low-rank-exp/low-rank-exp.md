# Speeding up Convolutional Neural Networks with Low Rank Expansions

## Scheme1

|     app      |  src-layer  |     tgt-layer      |   status   |
|:------------:|:-----------:|:------------------:|:----------:|
| `LowRankExp` | `nn.Conv2d` | `LowRankExpConvV1` | :confused: |

I meet some problems as follows:

### Problems

1. (**FATAL**) When using CUDA(2080Ti, cuda=11.0), `LowRankExpConvV1` inferences slower than `nn.Conv2d` whatever the number of base is; 
2. My implementation of filter reconstruction optimization is too slow to finish. (I use [CVXPY](https://www.cvxpy.org/) which seems to work on a single thread on CPU?)
   > UserWarning: Solution may be inaccurate. Try another solver, adjusting the solver settings, or solve with verbose=True for more information.

   ![filter reconstruction optimization](./scheme1-filter-opt.png)

### Experiments

I have just done some simple experiments to verify the correctness. I use an AlexNet model which has been trained on CIFAR10. There are some statistics.

|       model       | $\lambda$ | layers | #bases | macs(M) | params(M) | CPU time(ms) | CUDA time(ms) | acc@1(%) |
|:-----------------:|:--------:|:------:|:------:|:-------:|:---------:|:------------:|:-------------:|:--------:|
| AlexNet(Ordinary) |    -     |   -    |   -    | 698.89  |   44.43   |     1205     |     1.899     |  78.38   |
|     AlexNet-2     |  0.0001  |   2    |   8    | 556.31  |   44.23   |   964.955    |     2.425     |  76.01   |
|   AlexNet-2-sep   |  0.0001  |   2    |   8    | 550.33  |   44.22   |     1279     |     2.264     |  69.26   |

I only approximate the second convolutional layer of AlexNet(got AlexNet-2), which weight is a tensor of (64, 192, 3, 3). (In the order of (N, C, d, d)). I choose number of bases M = 48.
Then I use fixed $\lambda = 0.0001$ to optimize.

Maybe it is feasible to use a smaller M and ascending $\lambda$. But as the fatal problem say, this method seems meaningless compared to highly parallelized `torch.nn.Conv2d` on modern GPU.


