from colossalai.amp import AMP_TYPE
from model_zoo.gpt.gpt import GPTLMLoss
from model import GPT2_small_pipeline_hybrid
from torch.optim import Adam


BATCH_SIZE = 1
SEQ_LEN = 1024
NUM_EPOCHS = 60
NUM_MICRO_BATCHES = 1
PIPELINE = 2

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
    type=GPT2_small_pipeline_hybrid,
    checkpoint=True,
)

parallel = dict(
    pipeline=PIPELINE,
    tensor=dict(size=1, mode=None),
)
