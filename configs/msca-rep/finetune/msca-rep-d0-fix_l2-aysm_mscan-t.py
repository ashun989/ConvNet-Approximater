_base_ = ['../../_base_/models/mscan/mscan-t.py']

app = dict(
    type="MscaRep",
    decomp=0,
    fix=True
)

filters = []

hooks = [
    # dict(
    #     type='ClassEvalHook',
    #     priority=50,
    #     eval_cfg=dict(
    #         crop_pct=0.9,
    #         mean=(0.5, 0.5, 0.5),
    #         std=(0.5, 0.5, 0.5),
    #         data='/Zalick/Datasets/ILSVRC2012/', )
    # ),
    dict(
        type='L2Reconstruct',
        priority=50,
        asym=True,
        dataset_args=dict(
            root='/Zalick/Datasets/ILSVRC2012/',
            batch_size=64
        ),
        data_config=dict(
            crop_pct=0.9,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        optim_args=dict(
            opt='adamw',
            lr=1e-4,
            momentum=0.9,
            weight_decay=0.01,
        ),
        other_args=dict(
            log_interval=50,
            num_epochs=20
        )
    )
]

# output_name = 'mscan_t_d0_fix_ft.pth'
