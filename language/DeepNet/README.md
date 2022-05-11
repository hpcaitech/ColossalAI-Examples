# [DeepNet](https://arxiv.org/pdf/2203.00555.pdf): An Implementation based on [Colossal-AI](https://www.colossalai.org/)

## Overview

<p align="center">
  <img src="https://github.com/yuxuan-lou/ColossalAI-DeepNet/blob/main/IMG/overview.png" width="800">
</p>

This is the re-implementation of model DeepNet from paper [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/pdf/2203.00555.pdf).

DeepNet can scale transformer models to 1000 layers by applying DeepNorm. This Colossal-AI based implementation support data parallelism, pipeline parallelism and 1D tensor parallelism for training.

## How to prepare datasets

### Decoder-only DeepNet
The decoder-only DeepNet model is modified from the GPT model. In this example, we use WebText dataset for training. The way we prepare dataset is same as which in [Colossal-AI based GPT example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt).

## requirement

To use pipeline parallel training, you should install colossalai from the **latest** main branch.

## How to run

### Decoder-only DeepNet

```Bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json

colossalai run --nproc_per_node=<num_gpus> train_deepnet_decoder.py --config=decoder_configs/deepnet_pp1d.py
```


Please modify `DATA`, `num_gpus` with the path to your dataset and the number of GPUs respectively.
You can also modify the config `file decoder_configs/deepnet_pp1d.py` to further change parallel settings, training hyperparameters and model details.

## features

 - [x] Decoder-only DeepNet
 - [ ] Encoder-Decoder DeepNet
