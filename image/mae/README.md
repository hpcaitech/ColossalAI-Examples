# Train MAE on ImageNet 1000 (mini)

## Prepare Dataset

In the script, we used ImageNet 1000 (mini) dataset hosted on [kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/discussion).

Download and extract the dataset, then setting the environment variable `DATA`. 

```bash
export DATA=/path/to/data

# example
# ImageNet 1000 (mini) is extracted in the current directory
export DATA=$PWD/imagenet-mini/
```

## Run single-GPU training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node 1 train.py
```

