# Overview

Automatic Mixed Precision training provides significant arithmetic speedup by performing operations in half precision, and offers data transfer speedup by requiring less memory bandwidth. It also allocates less memory, enabling us to train larger models or train with larger batch size. In this example, we use one GPU to reproduce the training of Vision Transformer (ViT) on Caltech101 using colossalai. 

You may refer to [our documentation on mixed precision training](https://colossalai.org/tutorials/basic_tutorials/use_auto_mixed_precision_in_training) for more details.

# Prerequiste

```shell
pip install timm scipy
```

# How to run

On a single server, you can directly use `torch.distributed.launch` to start pre-training on multiple GPUs in parallel. In Colossal-AI, we provided fours ways to initialize the distributed environment. 

1. `colossalai.launch`
2. `colossalai.launch_from_torch`
3. `colossalai.launch_from_slurm`
4. `colossalai.launch_from_openmpi`

The first launch method is the most general for different cases and the remaining methods are helper methods for different launchers. In this example, we use `launch_from_torch` for demo purpose. 

The config file can be any file in the `config` directory:
- `config_AMP_apex.py`: rely on [Nvidia APEX](https://github.com/NVIDIA/apex) for mixed precision training
- `config_AMP_torch.py`: rely on [Torch CUDA AMP](https://pytorch.org/docs/stable/amp.html) for mixed precision training
- `config_AMP_naive.py`: all operations are performed in fp16
- `config_fp32.py`: all operations are performed in fp32


> ❗️ You should run with 1 GPU first if you do not have a ready dataset. If you run with 4 GPUs, the dataset will be downloaded and extracted simulateneously and may be corrupted.

You can invoke the following command to start training.

```shell
python -m torch.distributed.launch --nproc_per_node <world_size> --master_addr localhost --master_port 29500 train.py --config config/<config file>
```

For example, if you wish to run on 4 GPUs with Torch AMP.

```shell
python -m torch.distributed.launch --nproc_per_node 4 --master_addr localhost --master_port 29500 train.py --config config/config_AMP_torch.py
```

__

If you are using `colossalai.launch_from_slurm`, you can uncomment the `colossalai.launch_from_slurm` and comment out the `colossalai.launch_from_torch`.

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
colossalai.launch_from_slurm(config=args.config,
                            host=args.host,
                            port=args.port,
                            backend=args.backend
                            )
```

```shell
HOST=<node name> srun bash ./scripts/train_slurm.sh
```

---

# Experiments
In order to let everyone have a more intuitive feeling about amp, we use several amp methods to pretrain VIT-Base/16 on ImageNet-1K. The experimental results aims to prove that amp's efficiency in reducing memory and improving efficiency, so that hyperparameters such as learning rate may not be optimal.

|                  | RAM/GB | Iteration/s | throughput (batch/s) |
| ---------------- | ------ | ----------- | -------------------- |
| FP32 training    | 27.2   | 2.95        | 377.6                |
| AMP_TYPE.TORCH   | 20.5   | 3.25        | 416.0                |
| AMP_TYPE.NAIVE   | 17.0   | 3.53        | 451.8                |
| AMP_TYPE.APEX O1 | 20.2   | 3.07        | 393.0                |

As can be seen from the above table, the automixed precision training can reduces the RAM by 37.5% in the best cases, while increasing the throughput by 19.6%. Since the AMP reduces the memory cost for training models, we can further try enabling larger minibatches, which leads to larger throughput.


We also use the example code in this repo to train ViT-Base/16 on caltech101 dataset. The results are as follows:  

|                  | RAM/GB | Iteration/s | throughput (batch/s) |
| ---------------- | ------ | ----------- | -------------------- |
| FP32 training    | 25.0   | 0.84        | 107.5                |
| AMP_TYPE.TORCH   | 19.0   | 0.91        | 116.5                |
| AMP_TYPE.NAIVE   | 14.8   | 0.93        | 119.0                |
| AMP_TYPE.APEX O1 | 17.9   | 0.90        | 115.2                |

We observed a significant reduction in memory usage. The amp methods also slightly outperforms the full precision training in efficiency. However, the throughput of AMP training is not as well performed as in the last test(ImageNet-1K), which may be because the dataloader has become the bottleneck. It is very likely that most of the time is spent in reading data, and there is still a large computational advantage. You can add a timer to check the forward/backward time.




# Details
`config.py`
This is a [configuration file](https://colossalai.org/config.html) that defines hyperparameters and training scheme (fp16, gradient accumulation, etc.). The config content can be accessed through `gpc.config` in the program. By tuning the parallelism configuration, this example can be quickly deployed to a single server with several GPUs or to a large cluster with lots of nodes and GPUs. 


`train.py`
We start the training process using Colossal-AI.
