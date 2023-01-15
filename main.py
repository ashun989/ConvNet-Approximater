import torch
from torch import nn

from approx.models import AlexNet

if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)

    model = AlexNet()
    print(model)
    print(model(x).shape)

    model.classifier[-1] = nn.Identity()
    print(model)
    print(model(x).shape)
