
import colossalai
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.transformer.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def main():
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    logger = get_dist_logger()

    logger.info(f'GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])
    # build GPT model
    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(convert_fp16=True, target_device=torch.cuda.current_device(), shard_strategy=shard_strategy, shard_param=True):
        model = gpt2_medium(checkpoint=True)
    # Enable CPU offload for parameters and gradients
    model = ShardedModelV2(model, shard_strategy, offload_config={'device': 'cpu'})
    logger.info(f'GPU memory usage after init model: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    # Enable CPU offload for optimizer states
    optimizer = ShardedOptimizerV2(model, CPUAdam, cpu_offload=True, lr=1e-3)
    logger.info(f'GPU memory usage after init optim: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])

    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        model.train()
        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        optimizer.backward(loss)
        optimizer.step()
        logger.info(
            f'Step [{n+1}/{NUM_STEPS}] GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', ranks=[0])


if __name__ == '__main__':
    main()
