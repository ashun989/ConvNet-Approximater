# Speeding up Convolutional Neural Networks with Low Rank Expansions

## Scheme1

|     app      |  src-layer  |     tgt-layer      |   status   |
|:------------:|:-----------:|:------------------:|:----------:|
| `LowRankExp` | `nn.Conv2d` | `LowRankExpConvV1` | :confused: |

I meet some problems as follows:

### Problems

1. (**FATAL**) When using CUDA, `LowRankExpConvV1` inferences slower than `nn.Conv2d` whatever the number of base is; 
2. My implementation of filter reconstruction optimization is too slow to finish. (I use [CVXPY](https://www.cvxpy.org/) which seems to work on a single thread on CPU?)
   ![filter reconstruction optimization](./scheme1-filter-opt.png)