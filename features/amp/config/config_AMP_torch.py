from colossalai.amp import AMP_TYPE

# ViT Base
BATCH_SIZE = 128
DROP_RATE = 0.1
NUM_EPOCHS = 2

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

clip_grad_norm = 1.0
