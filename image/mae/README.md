# Pretrain MAE on ImageNet 1000 (mini)

Colossal-ai implementation of MAE, [arxiv](https//arxiv.org/abs/2111.06377).

As an example, we just cover the pretrain phase with ImageNet 1000
mini dataset. Helpers under subdir [util/](./util/) are from
[facebookresearch/deit](https://github.com/facebookresearch/deit),
under Apache License 2.0.

## Prepare Dataset

In the script, we used ImageNet 1000 (mini) dataset hosted on 
[kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/discussion).

Download and extract the dataset, then setting the environment
variable `DATA`, or soft link data to the default location `{config_dir}/data`

```bash
# example
export DATA=/path/to/imagenet-mini/

# or link to default place
ln -s /path/to/imagenet-mini/ ./data
```

## Run single-GPU training

This example is developed and tested under PyTorch 1.10, use `torchrun`
to run it:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node 1 main_pretrain.py
```

It would read [./config/pretrain.py](./config/pretrain.py) as startup
configuration, feel free to check it if you want to fine-tune the model
or get some insight.

By default, the pretrained model would generate a series of checkpoints, named
`./output/checkpoint-{epoch}.pth`.


## Run multi-GPU training

To run multi-GPU training on a single node, just change the `--nproc_per_node`
parameter. For example, if `--nproc_per_node=4`, 4 GPUs on this machine will be
used for training. However, to make sure the model converges well, you should 
adjust your batch size and learning rate accordingly.