# Vision Transformer with Hybrid Parallel

In this example, we will be running Vision Transformer on the ImageNet dataset with hybrid parallelism. 
Hybrid parallelism includes data, tensor and pipeline parallelism
We provided different configurations for you to execute different tensor parallelism including 1D, 2D, 2.5D and 3D tensor parallelism.

## How to Prepare Dataset

You can download the ImageNet dataset from the [ImageNet official website](https://www.image-net.org/download.php). You should get the raw images after downloading the dataset. As we use [DALI](https://github.com/NVIDIA/DALI) to read data, we use the TFRecords dataset instead of raw Imagenet dataset. This offers better speedup to IO. If you don't have TFRecords dataset, follow [imagenet-tools](https://github.com/ver217/imagenet-tools) to build one.
If you don't want to download the ImageNet, you can choose the cifar10 as your dataset, and use the train_with_cifar10 train script.

## How to Change the Configuration for Your Machines

We have put a list of configuration files in the `configs folder`. These 
configuration files are for at least 64 GPUs. 
You may not have this many GPUs, but that's fine. Below is a guide for you on how to change the parameters to adapt to your machine.

Firstly, we got to understand the size of each parallelism. The formula is that `number of GPUs = pipeline parallel size x tensor parallel size x data parallel size`.

Tensor and pipeline parallel size are defined in the `parallel` dictionary in the config file like below:

```python
parallel = dict(
    pipeline=<int>,
    tensor=dict(mode=<str>, size=<int>)
)
```
The data parallel size will be automatically calculated based on the formula above. 
Thus, you can modify the pipeline and tensor parallel size to make sure it fits your machine.

Secondly, you can modify the model configurations. As we benchmarked for large-scale models, the existing configuration for Vision Transformer is way larger than ViT-Large. You can change the parameters below to make it smaller if your machine cannot accommodate it.

```python
# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 4096
DEPTH = 32
NUM_HEADS = 64
MLP_RATIO = 4
NUM_CLASSES = 1000
CHECKPOINT = True
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE) ** 2 + 1  # add 1 for cls token

```

Lastly, you can change the hyperparameters for your training. 
One thing to note is that the `BATCH_SIZE` refers to local batch size on a GPU.
For example, if you have 32 GPUs, tensor parallel size 4 and pipeline size 4, your data parallel size will be `32 / 4 / 4 = 2`. Thus, you effective global batch size will be `2 x BATCH_SIZE`. 

```python
# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 300
WARMUP_EPOCHS = 32

```

## requirement

To use pipeline parallel training, you should install colossalai from the **latest** main branch.

## How to Run

Before you start training, you need to set the environment variable `DATA` so that the script knows where to fetch the data for DALI dataloader. If you do not know to make prepare the data for DALI dataloader, please jump to `How to Prepare ImageNet Dataset` section above.

```shell
export DATA=/path/to/ILSVRC2012
```

### Run on Multi-Node with SLURM
As this script runs on multiple nodes with 64 GPUs, we used SLURM scheduler to launch training. 
To execute training with `srun` provided by SLURM, you can use the following command.

```
# use engine
srun python train_with_engine.py --config ./configs/<config file> --host <node> --port 29500

# use trainer
srun python train_with_trainer.py --config ./configs<config file> --host <node> --port 29500
```

### Run on Single-Node with PyTorch Launcher

If you want to run experiments on a single node, you can use PyTorch Distributed Launcher Utility. First of all, you need to replace `colossalai.launch_from_slurm` with `colossalai.launch_from_torch`. The example code is as below.

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
colossalai.launch_from_torch(config=args.config)
```

In your terminal
```shell
colossalai run --nproc_per_node <world_size>  train.py --config config.py
```

### Using OpenMPI
If you use `OpenMPI` or other launchers, you can refer to [Launch Colossal-AI](https://colossalai.org/tutorials/basic_tutorials/launch_colossalai).

## Performance Tuning

Currently, we set number of micro batches for pipeline parallelism the same as the number of pipeline stages. 
This is only for benchmarking purpose. You can change this number in an attempt to obtain better throughput. 
This is only valid if your pipeline parallel size is larger than 1.

```python
NUM_MICRO_BATCHES = parallel['pipeline']`
```
