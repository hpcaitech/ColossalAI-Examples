from colossalai.amp import AMP_TYPE
from ..model.t5 import T5LMLoss, T5_small
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_EPOCHS = 60
TENSOR_PARALLEL = 4

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=T5LMLoss,
)

model = dict(
    type=T5_small,
    checkpoint=True,
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode='2d'),
)