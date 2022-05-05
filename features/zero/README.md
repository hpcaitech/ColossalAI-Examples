# ZeRO

## Prepare Model

In this example, we use `Hugging Face Transformers`. You have to install `transformers` before running this example. We will take `GPT2 Medium` as an example here.

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