_base_ = ['../_base_/models/alexnet/alexnet.py']

app = dict(
    type="LowRankExpV1",
    max_iter=10,
    min_lmda=0.0001,
    max_lmda=0.01,
    init_method='svd',
    lmda_length=10,
    num_bases=(8,),
    do_decomp=False
)

filters = [
    dict(
        type="SimpleConvFilter"
    ),
    dict(
        type="IndicesFilter",
        indices=(2,)
    )
]

hooks = [
    dict(
        type='ModelAnalysis',
        priority=50,
        input_shape=(3, 224, 224)
    ),
    dict(
        type='ClassEvalHook',
        priority=50,
        eval_cfg=dict(
            dataset='torch/cifar10',
            crop_pct=0.875,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation='bilinear',
            data='data'),
    ),
    dict(
        type='InferenceTimeHook',
        priority=50,
        infer_cfg=dict(
            input_size=(64, 3, 224, 224),
        )
    ),
]
