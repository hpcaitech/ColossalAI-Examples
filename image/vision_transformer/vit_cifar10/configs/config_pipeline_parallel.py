from colossalai.amp import AMP_TYPE


# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token

# parallel setting

parallel = dict(
    pipeline=2
)

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0


# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)