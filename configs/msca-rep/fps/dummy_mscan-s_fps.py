_base_ = ['../../_base_/models/mscan/mscan-s.py',
          '../../_base_/apps/dummy.py']

hooks = [
    dict(
        type='Fps',
        priority=50,
        repeat_times=3,
        dataset_args=dict(
            root='data/ILSVRC2012',
            batch_size=64
        )
    )
]