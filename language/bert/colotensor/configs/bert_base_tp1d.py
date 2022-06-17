SEQ_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1

parallel = dict(
    tensor=dict(mode="1d", size=4),
)

model = dict(
    type="bert_base",
)