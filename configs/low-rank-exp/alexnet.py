model = dict(
    type="AlexNet",
    checkpoint="pretrained/alexnet_cifar10.pth"
)

app = dict(
    type="LowRankExpV1",
    max_iter=5,
    min_lmda=0.001,
    max_lmda=1.0,
)
