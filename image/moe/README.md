# Overview

![Still In Progress](https://img.shields.io/badge/-Still%20In%20Progress-orange)

MoE is a new technique to enlarge neural networks while keeping the same throughput in our training. 
It is designed to improve the performance of our models without any additional time penalty. Our old 
version moe parallelism will cause a moderate computation overhead and additional memory usage. But
we are happy to announce that recently enabled CUDA kernels have solved the problem above. There 
are only two things that you need to concern. One is the additional communication time which highly
depends on the topology and bandwidth of the network in running environment. Another is extra memory usage,
since we have a larger model thanks to MoE. We will continuously maintain and optimize our MoE system
and be encouraged by any issue that can help us improve our system.

At present, we have provided Widenet and ViT-MoE in our model zoo (more information about Widenet can be 
found [here](https://arxiv.org/abs/2107.11817)). We now support a recent technique proposed by Microsoft, PR-MoE.
You can access [here](https://arxiv.org/abs/2201.05596) to know more about PR-MoE.
Directly use ViT-MoE in our model zoo or use MoeModule in your model to exploit PR-MoE.

Here is a simple example about how to run ViT-MoE Lite6 with PR-MoE on cifar10.

# How to run

Before running this training script, you must set a environment variable called `DATA` where you place
cifar10 data or want to place cifar10 data.

```shell
export DATA=<absolute path where you store cifar10 data> 
```

On a single server, you can directly use torchrun to start pre-training on multiple GPUs in parallel. 
If you use the script here to train, just use follow instruction in your terminal. `n_proc` is the 
number of processes which commonly equals to the number GPUs.

```shell
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --config ./config.py
```

If you want to use multi servers, please check our document about environment initialization.

Make sure to initialize moe running environment by `moe_set_seed` before building the model.

# Result

The best evaluation accuracy during training ViT-MoE Lite6 on cifar10 from scratch is 90.66%, which is better than average
performance in training ViT Lite7. The result can be improved by data augmentations such as mixup and Randaug.
We will offer those training scripts soon.