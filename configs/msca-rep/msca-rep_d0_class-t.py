model = dict(
    type="MSCAN_Classifier",
    init_cfg="pretrained/mscan_t_modified.pth",
    num_channels=(32, 64, 160, 256),
    num_blocks=(3, 3, 5, 2),
    exp_ratios=(8, 8, 4, 4),
    drop_rate=0.0,
    drop_path_rate=0.1
)

app = dict(
    type="MscaRep",
    decomp=0,
    fix=False
)

filters = []

hooks = [
    dict(
        type='ClassEvalHook',
        priority=50,
        eval_cfg=dict(
            crop_pct=0.9,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            data='/Zalick/Datasets/ILSVRC2012/', )
    )
]

output_name = 'mscan_t_d0.pth'
