model = dict(
    type="MSCAN_Classifier",
    init_cfg="pretrained/mscan_s_modified.pth",
    num_channels=(64, 128, 320, 512),
    num_blocks=(2, 2, 4, 2),
    exp_ratios=(8, 8, 4, 4),
    drop_rate=0.0,
    drop_path_rate=0.1
)