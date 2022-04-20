# BigGAN-Colossal-AI
This is a demo code of implementing BigGAN with Colossal-AI.

This repo contains code for 4 GPU training of BigGANs from [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) by Andrew Brock, Jeff Donahue, and Karen Simonyan.

## How To Use This Code
You will need:

- [PyTorch](https://PyTorch.org/), version 1.0.1
- Colossal-AI, version 0.1.2
- tqdm, numpy, scipy, and h5py
- The ImageNet training set

First, you may optionally prepare a pre-processed HDF5 version of your target dataset for faster I/O. Following this (or not), you'll need the Inception moments needed to calculate FID. These can both be done by modifying and running

```sh
sh scripts/utils/prepare_data.sh
```

Which by default assumes your ImageNet training set is downloaded into the root folder `data` in this directory, and will prepare the cached HDF5 at 128x128 pixel resolution.

With data prepared, you can run the training script as follows:

```sh
#!/bin/bash
LOCAL_RANK=<local rank>
NUM_NODES=<number of nodes>
NPROC_PER_NODE=<number of processes per node, i.e. GPUs per node>
MASTER_ADDR=<master address>
MASTER_PORT=<master port>

sh scripts/launch_BigGAN.sh
```


## Citation

If you use this code, please cite
```text
@inproceedings{
brock2018large,
title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
author={Andrew Brock and Jeff Donahue and Karen Simonyan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1xsqj09Fm},
}
```
```text
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```
