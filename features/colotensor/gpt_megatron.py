import colossalai
import psutil
import torch
import torch.nn as nn
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
from transformers import GPT2Config, GPT2LMHeadModel
from time import time
from functools import partial
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoParameter, ProcessGroup


def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    if param.process_group.tp_world_size() == 1:
        param.set_process_group(pg)
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                                n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len,
                                                vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

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


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def main():
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup(tp_degree=dist.get_world_size())
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    with ColoInitContext(device=torch.device('cpu')):
        model = gpt2_medium(checkpoint=True)
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'The original model size: {numel}', ranks=[0])

    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # set process group for all parameters
            param.set_process_group(pg)

            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)  # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)  # row slice
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)  # colmn slice
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)  # colmn slice

    model = model.cuda()
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'The tensor parallelism model size : {numel}', ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])
        loss.backward()
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])
        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        step_time = time() - start
        logger.info(
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}',
            ranks=[0])


if __name__ == '__main__':
    main()
