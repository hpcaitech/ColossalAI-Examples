from colossalai.amp import AMP_TYPE

# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 300
NUM_CLASSES = 10

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)


NUM_CHUNKS = 1
gradient_accumulation = 16
clip_grad_norm = 1.0