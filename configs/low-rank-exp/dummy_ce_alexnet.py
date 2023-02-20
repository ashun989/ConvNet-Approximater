_base_ = ['../_base_/models/alexnet/alexnet.py',
          '../_base_/apps/dummy.py']

hooks = [
    dict(
        type='L2Reconstruct',
        priority=50,
        asym=True,
        l2_weight=0.0,
        cls_weight=1.0,
        no_norm=True,
        layer_wise=False,
        dataset_args=dict(
            name='torch/cifar10',
            root='data',
            batch_size=64
        ),
        data_config=dict(
            crop_pct=0.875,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation='bilinear',
        ),
        optim_args=dict(
            opt='adamw',
            lr=1e-4,
            momentum=0.9,
            weight_decay=0.01,
        ),
        sche_args=dict(
            epochs=10
        ),
        other_args=dict(
            log_interval=300,
            layer_epochs=0,
            resume='',
            no_resume_opt=False,
            start_epoch=0
        )
    ),
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
    )
]
