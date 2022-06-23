# Vision Transformer with ColoTensor

# Overview

In this example, we will run Vision Transformer with ColoTensor.
We use model **ViTForImageClassification** from Hugging Face [Link](https://huggingface.co/docs/transformers/model_doc/vit).
You can change world size or decide whether use DDP in our code.

# Requirement

You should install colossalai from the **latest** main branch and install pytest, transformers with:

```shell
pip install pytest transformers
```

# How to run

In your terminal
```shell
pytest test_vit.py
```

This will evaluate models with different **world_size** and **use_ddp**.


'''shell
sh run.sh
'''

This will start Vit-S training with ImageNet.