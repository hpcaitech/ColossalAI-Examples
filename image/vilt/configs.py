from colossalai.amp import AMP_TYPE

BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 10

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
gradient_clipping = 1.0

parallel = dict(
    tensor=dict(size=2, mode='1d'),
)
num_epochs = 10

# config logging path
logging = dict(
    root_path='./logs'
)