from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from titans.model.gpt import gpt2_small

BATCH_SIZE = 2
NUM_EPOCHS = 60
SEQ_LEN = 1024


zero = dict(
    model_config=dict(
        tensor_placement_policy='cpu',
        shard_strategy=TensorShardStrategy(),
        reuse_fp16_shard=True
    ),
    optimizer_config=dict()
)


optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)
