from torch.optim import Adam
from colossalai.amp import AMP_TYPE
import torch
from titans.model.deepnet import deepnet_small
from titans.loss.lm_loss import GPTLMLoss

BATCH_SIZE = 8
NUM_EPOCHS = 2
SEQ_LEN = 1024

NUM_MICRO_BATCHES = 1
HIDDEN_SIZE = 768
PIPELINE = 2
TENSOR_PARALLEL = 2
MODE = '1d'
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

fp16 = dict(mode=AMP_TYPE.NAIVE)

parallel = dict(pipeline=PIPELINE, tensor=dict(mode=MODE, size=TENSOR_PARALLEL))

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=deepnet_small,
    checkpoint=True,
    dtype=torch.half,
)

loss = dict(type=GPTLMLoss,)
