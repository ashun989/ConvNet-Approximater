model = dict(
    type="AlexNet",
    init_cfg="pretrained/alexnet_cifar10.pth"
)

app = dict(
    type="LowRankExpV1",
    max_iter=5,
    min_lmda=0.0001,
    max_lmda=0.0001,
    lmda_length=1,
    # num_bases=(48, 144, 288, 192)
    num_bases=(12,)
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
