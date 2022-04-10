from colossalai.amp import AMP_TYPE

BATCH_SIZE = 4
NUM_EPOCHS = 2

CONFIG = dict(
    fp16=dict(
        mode=AMP_TYPE.TORCH
    )
)