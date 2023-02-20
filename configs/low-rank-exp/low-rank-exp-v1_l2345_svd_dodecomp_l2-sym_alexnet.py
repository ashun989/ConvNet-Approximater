_base_ = ['./low-rank-exp-v1_l2345_svd_dodecomp_alexnet.py']

layer_epochs = 2

hooks = [
    dict(
        type='L2Reconstruct',
        priority=50,
        asym=False,
        l2_weight=1.0,
        cls_weight=0.0,
        layer_wise=True,
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
            lr=1e-2,
            momentum=0.9,
            weight_decay=0.01,
        ),
        sche_args=dict(
            epochs=layer_epochs * 4
        ),
        other_args=dict(
            log_interval=300,
            layer_wise=True,
            layer_epochs=layer_epochs
        )
    )]
