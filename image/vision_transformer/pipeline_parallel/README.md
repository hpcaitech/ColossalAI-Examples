# Overview

This is an example to train `ViT` with pipeline parallel. As we just want to introduce you how to use pipeline parallel, we just use `CIFAR-10` dataset here for simplicity. 

# How to run

We use `colossalai.launch_from_torch` as an example here. Before running, you should `export DATA=/path/to/cifar`.

If you are training with single node multiple GPUs:
```shell
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <world_size> vit.py

# If your torch >= 1.9.0
python -m torch.distributed.run --standalone --nproc_per_node=<world_size> vit.py

# Otherwise
python -m torch.distributed.launch --nproc_per_node <world_size> --master_addr <node_name> --master_port 29500 vit.py
```

If you are using multiple nodes, see [torchrun](https://pytorch.org/docs/stable/elastic/run.html#launcher-api).
