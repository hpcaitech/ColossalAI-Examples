LOG_DIR = './logs'

SEQ_LENGTH = 128
BATCH_SIZE = 100
STEP_PER_EPOCH = 10
NUM_EPOCHS = 1000
WARMUP_PROPORTION = 0.01
LR = 7.8e-5

parallel = dict(
    tensor=dict(mode="1d", size=1),
)

model = dict(
    type="bert_base",
)