from colossalai.amp import AMP_TYPE

# your own global parameters
DATA_PATH = './data/text/my-bert_text_document'
VOCAB_FILE_PATH = './data/vocab/bert-large-uncased-vocab.txt'

# hyper parameters
TRAIN_ITERS = 10000
DECAY_ITERS = 9900
GLOBAL_BATCH_SIZE = 8
EVAL_ITERS = 10
EVAL_INTERVAL = 10
SEQ_LENGTH = 256
LR = 0.0001
MIN_LR = 1e-05
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.01

# BERT config
DEPTH = 12
NUM_ATTENTION_HEADS = 12
HIDDEN_SIZE = 768

# model config
ADD_BINARY_HEAD = False

# Colossal-AI parameter
fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=1, mode=None),
)









