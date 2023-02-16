_base_ = ['../_base_/models/mscan/mscan-t.py',
          '../_base_/apps/dummy.py']

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
        type='InferenceTimeHook',
        priority=50,
        infer_cfg=dict(
            input_size=(64, 3, 224, 224),
        )
    )
]