from colossalai.amp import AMP_TYPE

CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)
