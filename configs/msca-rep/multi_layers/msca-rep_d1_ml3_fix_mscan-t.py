_base_ = ['./msca-rep_d1_ml2_fix_mscan-t.py']

filters = [
    dict(
        type="IndicesFilter",
        indices=(1, 2, 3)
    )
]
