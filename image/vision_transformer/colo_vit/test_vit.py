import torch
import pytest
import colossalai
from colossalai.context.parallel_mode import ParallelMode
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, distspec
from colossalai.core import global_context as gpc
from functools import partial
from utils.util import tensor_equal, tensor_shard_equal, set_seed
from vit import get_training_components
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.parallel.data_parallel import ColoDDP


# Only for all Linear, it's 1d_row split because Linear will be transposed when calculating.
# But for other layers, it's 1d_col split.
# Layernorm is not supported for now.
# patch_embeddings.projection has nn.Conv2d
# https://github.com/huggingface/transformers/blob/dcb08b99f44919425f8ba9be9ddcc041af8ec25e/src/transformers/models/vit/modeling_vit.py#L182
def init_1d_row_for_linear_weight_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'layernorm' not in n and 'embeddings.patch_embeddings.projection.weight' not in n:
                p.set_spec(spec)


# Similarly, it's col split for Linear but row split for others.
def init_1d_col_for_linear_weight_bias_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n
                    or 'bias' in n) and 'layernorm' not in n and 'embeddings.patch_embeddings.projection' not in n:
                p.set_spec(spec)


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p, p)


def check_grad_equal(model, torch_model):
    for (np, p), (ntp, torch_p) in zip(model.named_parameters(), torch_model.named_parameters()):
        if (torch_p.grad.shape == p.grad.shape):
            assert torch.allclose(torch_p.grad, p.grad, rtol=1e-3, atol=2.0) == True
        else:
            dims_not_eq = torch.nonzero(torch.tensor(torch_p.grad.shape) != torch.tensor(p.grad.shape))
            dim = dims_not_eq.item()
            world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
            rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
            assert torch.allclose(torch_p.grad.chunk(world_size, dim)[rank], p.grad, rtol=1e-3, atol=2.0) == True


def run_vit(init_spec_func, use_ddp):
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_training_components()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda()
    torch_model = model_builder().cuda()
    if use_ddp:
        model = ColoDDP(model)
        torch_model = DDP(torch_model,
                          device_ids=[gpc.get_global_rank()],
                          process_group=gpc.get_group(ParallelMode.DATA))
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)
    init_spec_func(model)

    check_param_equal(model, torch_model)
    model.train()
    torch_model.train()
    set_seed(gpc.get_local_rank(ParallelMode.DATA))

    optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    torch_optimizer = optimizer_class(torch_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for i, image_dict in enumerate(train_dataloader):
        if use_ddp:
            model.zero_grad()
        else:
            optimizer.zero_grad()
        logits = model(image_dict['pixel_values'])
        torch_logits = torch_model(image_dict['pixel_values'])
        assert tensor_equal(torch_logits.logits, logits.logits)
        loss = criterion(logits.logits, image_dict['label'])
        torch_loss = criterion(torch_logits.logits, image_dict['label'])
        if use_ddp:
            model.backward(loss)
        else:
            loss.backward()
        torch_loss.backward()
        check_grad_equal(model, torch_model)
        optimizer.step()
        torch_optimizer.step()
        check_param_equal(model, torch_model)
        break


def run_dist(rank, world_size, port, use_ddp):
    if use_ddp and world_size == 1:
        return
    tp_world_size = world_size // 2 if use_ddp else world_size
    config = dict(parallel=dict(tensor=dict(mode="1d", size=tp_world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_vit(init_1d_row_for_linear_weight_spec, use_ddp)
    run_vit(init_1d_col_for_linear_weight_bias_spec, use_ddp)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_ddp', [False, True])
@rerun_if_address_is_in_use()
def test_vit(world_size, use_ddp):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), use_ddp=use_ddp)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_vit(4, False)
