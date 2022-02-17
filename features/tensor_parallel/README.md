# Tensor Parallelism Examples

## How to run

We recommend to use `torchrun` or `python -m torch.distribute.launch` to run examples here.

### 1D example
```shell
torchrun --nproc_per_node=2 tensor_parallel_1d.py --from_torch
```

### 2D example
```shell
torchrun --nproc_per_node=4 tensor_parallel_2d.py --from_torch
```

### 2.5D example
```shell
torchrun --nproc_per_node=8 tensor_parallel_2p5d.py --from_torch
```

### 3D example
```shell
torchrun --nproc_per_node=8 tensor_parallel_3d.py --from_torch
```