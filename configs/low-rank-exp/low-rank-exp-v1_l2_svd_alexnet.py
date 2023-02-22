_base_ = ['../_base_/models/alexnet/alexnet.py',
          './low-rank-exp-v1_l2_alexnet.py']

app = dict(
    type="LowRankExpV1",
    max_iter=0,
    min_lmda=0,
    max_lmda=0,
    init_method='svd',
    lmda_length=1,
    num_bases=(8,),
    do_decomp=False,
    init_decomp=False
)

