# Train ResNet50 on CIFAR10 with pipeline

## requirement

To use pipeline parallel training, you should install colossalai from the **latest** main branch.
## How to run

We use `colossalai.launch_from_torch` as an example here. Before running, you should `export DATA=/path/to/cifar`. 

If you are training with single node multiple GPUs:
```shell
colossalai run --nproc_per_node <world_size> resnet.py
```
