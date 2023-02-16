_base_ = ['./msca-rep_d1_l1_fix_class-t.py']

filters = [
    dict(
        type="IndicesFilter",
        indices=(10,)
    )
]


output_name = 'mscan_t_d1_l10_fix.pth'
