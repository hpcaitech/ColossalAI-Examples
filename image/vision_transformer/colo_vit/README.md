# Vision Transformer with ColoTensor

# Overview

In this example, we will run Vision Transformer with ColoTensor.

We use model **ViTForImageClassification** from Hugging Face [Link](https://huggingface.co/docs/transformers/model_doc/vit) for unit test.
You can change world size or decide whether use DDP in our code.

We use model **vision_transformer** from timm [Link](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) for training example.

(2022/6/28) The default configuration now supports 2DP+2TP with gradient accumulation and checkpoint support. Zero is not supported at present.

# Requirement

You should install colossalai from the **latest** main branch.

## Unit test
To run unit test, you should install pytest, transformers with:
```shell
pip install pytest transformers
```

## Training example
To run training example with ViT-S, you should install **NVIDIA DALI** from [Link](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html), timm and titans with:
```shell
pip install timm titans
```


# How to run

## Unit test
In your terminal
```shell
pytest test_vit.py
```

This will evaluate models with different **world_size** and **use_ddp**.

## Training example
Modify the settings in run.sh according to your environment, then in your terminal 
'''shell
sh run.sh
'''

This will start ViT-S training with ImageNet.