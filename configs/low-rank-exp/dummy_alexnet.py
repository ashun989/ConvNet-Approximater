_base_ = ['../_base_/models/alexnet/alexnet.py',
          '../_base_/apps/dummy.py']

hooks = [
    dict(
        type='InferenceTimeHook',
        priority=50,
        infer_cfg=dict(
            input_size=(4, 3, 224, 224),
        )
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
]
