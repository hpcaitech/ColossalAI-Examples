import math
import os
import time

import colossalai
import numpy as np
import torch
from colossalai import nn as col_nn
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.utils import get_current_device, print_rank_0
from torch import dtype, nn
from tqdm import tqdm
from colossalai.context import ParallelMode
from colossalai.communication import all_reduce


class ViTEmbedding1D(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embedding_dim: int,
        dropout: float,
        dtype: dtype = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_embed = col_nn.VanillaPatchEmbedding(img_size,
                                                        patch_size,
                                                        in_chans,
                                                        embedding_dim,
                                                        dtype=dtype,
                                                        flatten=flatten)
        self.dropout = col_nn.Dropout1D(dropout)
        print_rank_0(f'Patch embedding: parallel activation = {env.parallel_input_1d}.')

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x


class ViTSelfAttention1D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float,
        dropout: float,
        bias: bool = True,
        dtype: dtype = None,
    ):
        super().__init__()
        self.attention_head_size = dim // num_heads
        self.query_key_value = col_nn.Linear1D_Col(dim, 3 * dim, dtype=dtype, bias=bias, gather_output=False)
        print_rank_0(f'Self attention: parallel activation after the 1st linear layer = {env.parallel_input_1d}.')
        self.attention_dropout = col_nn.Dropout1D(attention_dropout)
        self.dense = col_nn.Linear1D_Row(dim, dim, dtype=dtype, bias=bias, parallel_input=env.parallel_input_1d)
        print_rank_0(f'Self attention: parallel activation after the 2nd linear layer = {env.parallel_input_1d}.')
        self.dropout = col_nn.Dropout1D(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size, )
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x


class ViTMLP1D(nn.Module):

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        dropout: float,
        dtype: dtype = None,
        bias: bool = True,
    ):
        super().__init__()
        self.dense_1 = col_nn.Linear1D_Col(dim, mlp_ratio * dim, dtype=dtype, bias=bias, gather_output=False)
        print_rank_0(f'MLP: parallel activation after the 1st linear layer = {env.parallel_input_1d}.')
        self.activation = nn.functional.gelu
        self.dropout_1 = col_nn.Dropout1D(dropout)
        self.dense_2 = col_nn.Linear1D_Row(mlp_ratio * dim,
                                           dim,
                                           dtype=dtype,
                                           bias=bias,
                                           parallel_input=env.parallel_input_1d)
        print_rank_0(f'MLP: parallel activation after the 2nd linear layer = {env.parallel_input_1d}.')
        self.dropout_2 = col_nn.Dropout1D(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x


class ViTBlock1D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        attention_dropout: float = 0.,
        dropout: float = 0.,
        dtype: dtype = None,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=1e-6).to(dtype).to(get_current_device())
        self.attn = ViTSelfAttention1D(dim=dim,
                                       num_heads=num_heads,
                                       attention_dropout=attention_dropout,
                                       dropout=dropout,
                                       bias=bias,
                                       dtype=dtype)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=1e-6).to(dtype).to(get_current_device())
        self.mlp = ViTMLP1D(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout, dtype=dtype, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHead1D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_classes: int,
        dtype: dtype = None,
        bias: bool = True,
    ):
        super().__init__()
        self.dense = col_nn.Classifier1D(dim, num_classes, dtype=dtype, bias=bias)
        print_rank_0(f'Head: parallel activation = {env.parallel_input_1d}.')

    def forward(self, x):
        x = x[:, 0]
        x = self.dense(x)
        return x


class ViT_Lite_1D(nn.Module):

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        depth: int = 7,
        num_heads: int = 4,
        dim: int = 256,
        mlp_ratio: int = 2,
        attention_dropout: float = 0.,
        dropout: float = 0.1,
        dtype: dtype = None,
        bias: bool = True,
    ):
        super().__init__()

        embed = ViTEmbedding1D(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=in_chans,
                               embedding_dim=dim,
                               dropout=dropout,
                               dtype=dtype)

        blocks = [
            ViTBlock1D(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                dtype=dtype,
                bias=bias,
            ) for _ in range(depth)
        ]

        norm = nn.LayerNorm(normalized_shape=dim, eps=1e-6).to(dtype).to(get_current_device())

        head = ViTHead1D(dim=dim, num_classes=num_classes, dtype=dtype, bias=bias)

        self.layers = nn.Sequential(
            embed,
            *blocks,
            norm,
            head,
        )

    def forward(self, x):
        x = self.layers(x)
        return x


CONFIG = dict(parallel=dict(
    pipeline=1,
    tensor=dict(size=2, mode='1d'),
))

DATASET_PATH = str(os.environ['DATA'])


def build_cifar(batch_size):
    import torchvision
    from colossalai.utils import get_dataloader
    from torchvision import transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader


def train(epoch, train_dataloader, model, criterion, optimizer, lr_scheduler):

    def mixup_data(x, y, alpha):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(get_current_device())

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        lam = torch.tensor([lam]).to(mixed_x.dtype).to(get_current_device())
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    model.train()
    progress = range(len(train_dataloader))
    if gpc.get_global_rank() == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = torch.zeros(()).to(torch.float).to(get_current_device())
    used_time = 0.
    num_steps = 0
    num_samples = 0
    data_iter = iter(train_dataloader)

    for _ in progress:
        fwd_start = time.time()

        inputs, targets = next(data_iter)
        inputs = inputs.to(get_current_device()).detach()
        targets = targets.to(get_current_device()).detach()
        batch_size = targets.size(0)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.8)

        outputs = model(inputs)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()

        fwd_end = time.time()

        bwd_start = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        bwd_end = time.time()

        num_steps += 1
        num_samples += batch_size

        fwd_time = fwd_end - fwd_start
        bwd_time = bwd_end - bwd_start
        used_time += fwd_time + bwd_time

        if gpc.get_global_rank() == 0:
            progress.set_postfix(loss=loss.item(),
                                 lr=lr_scheduler.get_lr()[0],
                                 time_forward=fwd_time,
                                 time_backward=bwd_time,
                                 throughput=batch_size * gpc.data_parallel_size / (fwd_time + bwd_time + 1e-12))

    train_loss = all_reduce(train_loss, ParallelMode.DATA)

    print_rank_0(f'[Epoch {epoch} / Train]: Loss = {train_loss.item() / (gpc.data_parallel_size * num_steps):.3f} | ' +
                 f'Throughput = {num_samples * gpc.data_parallel_size / (used_time + 1e-12):.3f} samples/sec')


def test(epoch, test_dataloader, model, criterion, accuracy):
    model.eval()
    progress = range(len(test_dataloader))
    if gpc.get_global_rank() == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Test]")

    test_loss = torch.zeros(()).to(torch.float).to(get_current_device())
    test_correct = torch.zeros(()).to(torch.int).to(get_current_device())
    used_time = 0.
    num_steps = 0
    num_samples = 0
    data_iter = iter(test_dataloader)

    for _ in progress:
        batch_start = time.time()

        inputs, targets = next(data_iter)
        inputs = inputs.to(get_current_device()).detach()
        targets = targets.to(get_current_device()).detach()
        batch_size = targets.size(0)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss

        acc = accuracy(outputs, targets)
        test_correct += acc

        batch_end = time.time()

        num_steps += 1
        num_samples += batch_size

        batch_time = batch_end - batch_start
        used_time += batch_time

        if gpc.get_global_rank() == 0:
            progress.set_postfix(loss=loss.item(),
                                 step_time=batch_time,
                                 accuracy=acc.item() * 100 / batch_size,
                                 throughput=batch_size * gpc.data_parallel_size / (batch_time + 1e-12))

    test_loss = all_reduce(test_loss, ParallelMode.DATA)
    test_correct = all_reduce(test_correct, ParallelMode.DATA)

    print_rank_0(f'[Epoch {epoch} / Test]: Loss = {test_loss.item() / (gpc.data_parallel_size * num_steps):.3f} | ' +
                 f'Accuracy = {test_correct.item() * 100 / (gpc.data_parallel_size * num_samples):.3f} % | ' +
                 f'Throughput = {num_samples * gpc.data_parallel_size / (used_time + 1e-12):.3f} samples/sec')


def train_cifar():
    from colossalai.logging import disable_existing_loggers

    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    if args.from_torch:
        colossalai.launch_from_torch(config=CONFIG, seed=42)
    else:
        colossalai.launch(config=CONFIG,
                          rank=args.rank,
                          world_size=args.world_size,
                          local_rank=args.local_rank,
                          host=args.host,
                          port=args.port,
                          seed=42)

    BATCH_SIZE = 512
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 40

    train_dataloader, test_dataloader = build_cifar(BATCH_SIZE // gpc.data_parallel_size)

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(ViT_Lite_1D(), process_group=gpc.get_group(ParallelMode.DATA))

    criterion = nn.CrossEntropyLoss(reduction='mean')

    accuracy = col_nn.Accuracy()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-2)

    steps_per_epoch = len(train_dataloader)

    lr_scheduler = col_nn.CosineAnnealingWarmupLR(optimizer=optimizer,
                                                  total_steps=NUM_EPOCHS * steps_per_epoch,
                                                  warmup_steps=WARMUP_EPOCHS * steps_per_epoch)

    for epoch in range(NUM_EPOCHS):
        train(epoch, train_dataloader, model, criterion, optimizer, lr_scheduler)
        test(epoch, test_dataloader, model, criterion, accuracy)


if __name__ == '__main__':
    train_cifar()
