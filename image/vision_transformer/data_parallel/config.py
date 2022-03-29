from colossalai.amp import AMP_TYPE

# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 2

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
clip_grad_norm = 1.0

dali = dict(
    gpu_aug=True,
    mixup_alpha=0.2
)
