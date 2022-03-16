from colossalai.amp import AMP_TYPE

# your own global parameters
DATA_PATH = './data/text/my-bert_text_document'
VOCAB_FILE_PATH = './data/vocab/bert-large-uncased-vocab.txt'

# hyper parameters
TRAIN_ITERS = 10000
GLOBAL_BATCH_SIZE = 8
EVAL_ITERS = 10
EVAL_INTERVAL = 10
SEQ_LENGTH = 256



# +
# Colossal-AI parameter
# -

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=1, mode=None),
)









