from colossalai.amp import AMP_TYPE


CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    ),
    parallel=dict(
        data=1,
        pipeline=1,
        tensor=dict(size=4, mode='2d'),
    )
)
