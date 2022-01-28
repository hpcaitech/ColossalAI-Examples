from colossalai.amp import AMP_TYPE

BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 200

fp16 = dict(
    mode=None,
)

zero = dict(
    level=3,
    verbose=False,
    offload_optimizer_config=dict(
        device='cpu',
        pin_memory=True,
        buffer_count=5,
        fast_init=False
    ),
    offload_param_config=dict(
        device='cpu',
        pin_memory=True,
        buffer_count=5,
        buffer_size=1e8,
        max_in_cpu=1e9
    )
)

parallel = dict(
    pipeline=dict(size=1),
    tensor=dict(size=1, mode=None)
)
