model = dict(
    type="AlexNet",
    checkpoint="pretrained/alexnet_cifar10.pth"
)

app = dict(
    type="LowRankExpV1",
    max_iter=3,
    min_lmda=0.0001,
    max_lmda=0.1,
    lmda_length=1,
    speed_ratio=3
)

filters = [
    dict(
        type="SimpleConvFilter"
    ),
    dict(
        type="IndicesFilter",
        indices=(3,)
    )
]
