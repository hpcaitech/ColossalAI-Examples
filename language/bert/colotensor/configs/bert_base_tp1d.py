LOG_DIR = './logs'

SEQ_LENGTH = 512
BATCH_SIZE = 8
STEP_PER_EPOCH = 10
NUM_EPOCHS = 100
WARMUP_EPOCHS = 2

parallel = dict(
    tensor=dict(mode="1d", size=2),
)

model = dict(
    type="bert_base",
)