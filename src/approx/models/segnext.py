# from . import SwitchableModel, MODEL
# from .mscan import MSCAN
#
# from torch import nn
#
# @MODEL.register_module()
# class MSCAN_Segmentor(SwitchableModel):
#     def __init__(self, in_channels=3,
#                  num_channels=(32, 64, 160, 256),
#                  num_blocks=(3, 3, 5, 2),
#                  exp_ratios=(8, 8, 4, 4),
#                  drop_rate=0.,
#                  drop_path_rate=0.,
#                  num_classes=1000,
#                  init_cfg=None
#                  ):
#         super(MSCAN_Classifier, self).__init__(init_cfg=init_cfg)
#         self.num_classes = num_classes
#         self.backbone = MSCAN(in_channels=in_channels,
#                               num_channels=num_channels,
#                               num_blocks=num_blocks,
#                               exp_ratios=exp_ratios,
#                               drop_rate=drop_rate,
#                               drop_path_rate=drop_path_rate)
#
#
#     def forward(self, x):
#         features = self.backbone(x)
