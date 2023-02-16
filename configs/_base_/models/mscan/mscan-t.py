model = dict(
    type="MSCAN_Classifier",
    init_cfg="pretrained/mscan_t_modified.pth",
    num_channels=(32, 64, 160, 256),
    num_blocks=(3, 3, 5, 2),
    exp_ratios=(8, 8, 4, 4),
    drop_rate=0.0,
    drop_path_rate=0.1
)