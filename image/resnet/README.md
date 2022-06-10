# Train ResNet on CIFAR10

## Prepare Dataset

We use CIFAR10 dataset in this example. The dataset will be downloaded to `./data` by default. 
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```


## Run single-GPU training

We provide two examples of training resnet 18 on the CIFAR10 dataset. You can choose other ResNet models in `resnet.py` as well.
You can change the value of `nproc_per_node` to adjust the number of GPUs used for training. 
When the `nproc_per_node` is changed, you may need to adjust the learning rate and batch size in the `config.py` accordingly.
Normally we follow the rule of linear scaling, which is `new_global_batch_size / new_learning_rate = old_global_batch_size / old_learning rate`.

```bash
# with engine
colossalai run --nproc_per_node 1 train.py

# with trainer
colossalai run --nproc_per_node 1 train.py --use_trainer
```

## Experiment Results

| model      | dataset     | Testing Accuracy |
| -          | -           | -                |
| ResNet18   | CIFAR10     | 95.2%            |
