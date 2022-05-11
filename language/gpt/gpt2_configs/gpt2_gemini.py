from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model_zoo.gpt import gpt

BATCH_SIZE = 8
NUM_EPOCHS = 60
SEQ_LEN = 1024


# parallel = dict(
#     tensor=dict(mode='2d', size=4)
# )

zero = dict(
    model_config=dict(
        tensor_placement_policy='cpu',
        shard_strategy=TensorShardStrategy(),
        reuse_fp16_shard=True
    ),
    optimizer_config=dict(initial_scale=2**5, gpu_margin_mem_ratio=0.2)
)


optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt.gpt2_8B,
    checkpoint=True,
)
