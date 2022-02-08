from colossalai.amp import AMP_TYPE
from model_zoo.gpt.gpt import GPTLMLoss, gpt2_small
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_EPOCHS = 60
TERSOR_PARALLEL = 8
DEPTH = 2


optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=GPTLMLoss,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)


parallel = dict(
    pipeline=1,
    tensor=dict(size=TERSOR_PARALLEL, depth=DEPTH, mode='2.5d'),
)
