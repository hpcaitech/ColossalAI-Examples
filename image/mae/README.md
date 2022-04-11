# Train MAE on ImageNet 1000 (mini)

## Prepare Dataset

In the script, we used ImageNet 1000 (mini) dataset hosted on [kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/discussion).

Download and extract the dataset, then setting the environment variable `DATA`, or soft link data to the default location `{config_dir}/data`

```bash
# example
export DATA=/path/to/imagenet-mini/

# or link to default place
ln -s /path/to/imagenet-mini/ ./data
```

## Run single-GPU training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node 1 train.py
```

