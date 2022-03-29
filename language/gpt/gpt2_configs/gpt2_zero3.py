from colossalai.nn.optimizer import CPUAdam
from colossalai.nn.optimizer.fused_adam import FusedAdam
from torch.optim.adam import Adam
from colossalai.nn.optimizer.hybrid_adam import HybridAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model_zoo.gpt.gpt import gpt2_small

BATCH_SIZE = 64
NUM_EPOCHS = 60
SEQ_LEN = 1024


zero = dict(
    model_config=dict(
        #offload_config=dict(device="cpu"),
        shard_strategy=TensorShardStrategy()
    ),
    #optimizer_config=dict(
        #cpu_offload=True,
    #)
)


optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)
