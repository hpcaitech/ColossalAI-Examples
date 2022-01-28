from colossalai.amp import AMP_TYPE

BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 200

fp16 = dict(
    mode=None,
)

zero = dict(
    level=2,
    cpu_offload=True,
    verbose=False,
)

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None)
)
