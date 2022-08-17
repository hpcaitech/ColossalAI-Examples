# Use tensor model paralelism via ColoTensor

## Introduction

This is an example of the turorial, **Parallelize Your Training like Megatron-LM via ColoTensor**.
It can tell you how to make your model adapted to tensor model parallelism.
Just use the below code to run the example.

```bash
colossalai run --nproc_per_node <world_size> gpt_megatron.py
```