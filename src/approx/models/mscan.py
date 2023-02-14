import torch
import torch.nn as nn

from approx.layers import DropPath, ParallelConv, LAYER
from approx.models import MODEL, SwitchableModel


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.proj(x)


class FFN(nn.Module):
    def __init__(self,
                 num_channel,
                 hidden_channel,
                 drop):
        super(FFN, self).__init__()
        self.fc1 = nn.Conv2d(
            in_channels=num_channel,
            out_channels=hidden_channel,
            kernel_size=1
        )

        self.dconv = nn.Conv2d(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channel,
        )

        self.fc2 = nn.Conv2d(
            in_channels=hidden_channel,
            out_channels=num_channel,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


@LAYER.register_module()
class MSCA(nn.Module):
    def __init__(self,
                 num_channel,
                 k1_size,
                 k_sizes):
        super(MSCA, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel,
                               kernel_size=k1_size, padding=k1_size // 2, groups=num_channel)

        paddings = [k // 2 for k in k_sizes]
        self.sd_convs = ParallelConv(num_channel, k_sizes, paddings, len(k_sizes), True, True)
        self.channel_mix = nn.Conv2d(num_channel, num_channel, 1)

    def forward(self, x):
        return x * self.channel_mix(self.sd_convs(self.conv0(x)))


class SpatialAttention(nn.Module):
    def __init__(self, num_channel, k1_size=5, k_sizes=(7, 11, 21)):
        super(SpatialAttention, self).__init__()
        self.proj_1 = nn.Conv2d(num_channel, num_channel, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSCA(num_channel, k1_size, k_sizes)
        self.proj_2 = nn.Conv2d(num_channel, num_channel, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MultiScaleConvAttnModule(nn.Module):
    """
    A Stage of MSCAN
    """

    def __init__(self,
                 num_channel,
                 hidden_channel,
                 drop,
                 drop_path
                 ):
        super(MultiScaleConvAttnModule, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_channel)
        self.attn = SpatialAttention(num_channel=num_channel)
        self.norm2 = nn.BatchNorm2d(num_channel)
        self.mlp = FFN(
            num_channel=num_channel,
            hidden_channel=hidden_channel,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((num_channel)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((num_channel)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        return x


class DownSample(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super(DownSample, self).__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.norm(self.proj(x))


class MSCAN(nn.Module):
    """The Backbone
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(32, 64, 160, 256),
                 num_blocks=(3, 3, 5, 2),
                 exp_ratios=(8, 8, 4, 4),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 ):
        super().__init__()

        assert len(num_channels) == len(num_blocks) == len(exp_ratios)

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.exp_rations = exp_ratios
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.num_blocks))]
        cur = 0

        for i, num_block in enumerate(self.num_blocks):
            out_c = self.num_channels[i]
            hid_c = self.num_channels[i] * self.exp_rations[i]
            downsample = StemConv(in_channels, num_channels[0]) if i == 0 \
                else DownSample(num_channels[i - 1], out_c)
            stage = nn.Sequential(
                *[MultiScaleConvAttnModule(
                    num_channel=out_c,
                    hidden_channel=hid_c,
                    drop=drop_rate,
                    drop_path=dpr[cur + j]
                ) for j in range(num_block)]
            )
            norm = nn.LayerNorm(out_c)  # same as ViT
            self.layers.append(nn.ModuleList([downsample, stage, norm]))
            cur += num_block

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer[0](x)  # down
            x = layer[1](x)  # stage
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
            x = layer[2](x)  # layer norm
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features

    # def init_weights(self, pretrained=None):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m, mode='fan_in', bias=0.)
    #         elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
    #             constant_init(m, val=1.0, bias=0.)


@MODEL.register_module()
class MSCAN_Classifier(SwitchableModel):
    def __init__(self, in_channels=3,
                 num_channels=(32, 64, 160, 256),
                 num_blocks=(3, 3, 5, 2),
                 exp_ratios=(8, 8, 4, 4),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_classes=1000,
                 init_cfg=None
                 ):
        super(MSCAN_Classifier, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.backbone = MSCAN(in_channels=in_channels,
                              num_channels=num_channels,
                              num_blocks=num_blocks,
                              exp_ratios=exp_ratios,
                              drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(num_channels[-1], self.num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)[-1]
        return self.head(self.gap(x).squeeze(-1).squeeze(-1))
