_base_ = ['../_base_/models/alexnet/alexnet.py',
          './low-rank-exp-v1_l2345_svd_alexnet.py']

app = dict(
    init_decomp=True
)

hooks = [
    dict(
        type="CkptHook",
        priority=5,
        ckpt_cfg=dict(
            after_initialize=dict(
                action="load",
                path="work_dir/lre-v1_l2345_svd_dodecomp_alexnet/20230222182847/lre-v1_l2345_svd_dodecomp_alexnet_add-sub.pth"
            )
        )
    )
]
