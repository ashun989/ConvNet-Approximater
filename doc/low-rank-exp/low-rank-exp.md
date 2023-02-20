# Speeding up Convolutional Neural Networks with Low Rank Expansions

## Scheme1

|     app      |  src-layer  |     tgt-layer      |   status   |
|:------------:|:-----------:|:------------------:|:----------:|
| `LowRankExp` | `nn.Conv2d` | `LowRankExpConvV1` | :confused: |

![filter reconstruction optimization](./scheme1-filter-opt.png)

### Experiments

I have just done some simple experiments to verify the correctness. I use an AlexNet model which has been trained on
CIFAR10. There are some statistics. Inference time is tested with random input of `(64, 3, 224, 224)`


|       model       | init  | max iters | $\lambda$ | layers  | #bases  | macs(M) | params(M) | CPU time(ms) | CUDA time(ms) | acc@1(%) |
|:-----------------:|:-----:|:---------:|:---------:|:-------:|:-------:|:-------:|:---------:|:------------:|:-------------:|:--------:|
| AlexNet(Ordinary) |   -   |     -     |     -     |    -    |    -    | 698.89  |   44.43   |     1265     |     6.605     |  78.38   |
|   AlexNet-2345    | 'svd' |     0     |     0     | 2,3,4,5 | 8,8,8,8 | 527.41  |   44.06   |     1279     |    11.632     |  76.31   |
| AlexNet-2345-sep  | 'svd' |     0     |     0     | 2,3,4,5 | 8,8,8,8 | 516.93  |   44.03   |     1270     |    14.439     |  69.85   |




