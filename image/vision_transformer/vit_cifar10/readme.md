# ViT training on Cifar10

## Overview
Here is an example of training vision transformer on Cifar10 dataset with Colossal-AI. It supports data parallel, tensor parallel and pipeline paralle.

## Prepare data
Since Cifar10 is a relatetively small dataset, it is not necessary to prepare data beforehead. During training process, Cifar10 dataset will be automatically downloaded to `DATA` path.

## How to run



### Data Parallel

```
export DATA=<path_to_data>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_dp.py --config ./configs/config_data_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_dp.py --config ./configs/config_data_parallel.py
# Otherwise
# python -m torch.distributed.launch --nproc_per_node <NUM_GPUs> --master_addr <node_name> --master_port 29500 train_dp.py --config ./configs/config.py
```

`DATA` is where Cifar10 dataset will be automatically downloaded and saved.

### Pipeline Parallel
```
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_pipeline_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_pipeline_parallel.py
```


### Hybrid Parallel
```
export DATA=<path_to_dataset>
# If your torch >= 1.10.0
torchrun --standalone --nproc_per_node <NUM_GPUs>  train_hybrid.py --config ./configs/config_hybrid_parallel.py
# If your torch >= 1.9.0
# python -m torch.distributed.run --standalone --nproc_per_node= <NUM_GPUs> train_hybrid.py --config ./configs/config_hybrid_parallel.py
```
