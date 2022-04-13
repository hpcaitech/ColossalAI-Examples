# Train ResNet18 on Compound property prediction

Compound property prediction examples with resnet 

##  Dataset


Please download [dataset](https://drive.google.com/file/d/1Sgkc4UX8bRoAdTvNrCCvUgLd7T_zjaI-/view?usp=sharing) and zip it.




## Run single-GPU training

We provide two examples of training resnet 34 on the CIFAR10 dataset. One example is with engine and the other is 
with the trainer. You can invoke the training script by the following command. This batch size and learning rate 
are for a single GPU. Thus, in the following command, `nproc_per_node` is 1, which means there is only one process 
invoked. If you change `nproc_per_node`, you will have to change the learning rate accordingly as the global batch
size has changed.

```bash
# with engine
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 run_resnet_property_prediction_with_engine.py

# with trainer
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 run_resnet_property_prediction_with_trainer.py
```

If you have PyTorch version >= 1.10, you can use `torchrun` instead.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node 1 run_resnet_property_prediction_with_engine.py

torchrun --standalone --nnodes=1 --nproc_per_node 1 run_resnet_property_prediction_with_trainer.py
```

## Run multi-GPU training

To run multi-GPU training on a single node, just change the `--nproc_per_node` parameter. For example, if `--nproc_per_node=4`, 4 GPUs on this machine will be
used for training. However, to make sure the model converges well, you should adjust your batch size and learning rate accordingly.
