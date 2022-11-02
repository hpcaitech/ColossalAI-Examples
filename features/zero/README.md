# ZeRO

## Prepare Model

In this example, we use `Hugging Face Transformers`. You have to install `transformers` before running this example. We will take `GPT2 Medium` as an example here. train.py is an old API before ColossalAI v0.1.9, while train_v2.py is the lastest API after v0.1.10.

```shell
# install huggingface transformers
pip install transformers
```

## Prepare Data

This example is intended for showing you how to use `ZeRO`. For simplicity, we just use randomly generated data here.

## Run with ZeRO

We just use naive training loop in this example. `Engine` and `Trainer` are not used.

```shell
colossalai run --nproc_per_node=1 train.py
```
