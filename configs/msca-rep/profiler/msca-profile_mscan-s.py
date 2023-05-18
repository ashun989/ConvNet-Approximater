_base_ = ['../../_base_/models/mscan/mscan-s.py']

app = dict(type="MscaProfile")

hooks = [
    dict(
        type='InferenceTimeHook',
        priority=50,
        infer_cfg=dict(
            input_size=(64, 3, 224, 224),
            table_args=dict(sort_by='cuda_time_total')
            # profile_args=dict(with_stack=True),
            # key_args=dict(group_by_stack_n=5),
        )
    ),
]