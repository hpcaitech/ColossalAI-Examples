from colossalai.amp import AMP_TYPE


CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    ),
    parallel=dict(
        pipeline=1,
        tensor=dict(size=8, mode='3d'),
    )
)
