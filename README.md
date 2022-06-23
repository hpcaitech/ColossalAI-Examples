# ColossalAI-Examples

## Introduction

This repository provides various examples for **Colossal-AI**. For each feature of 
Colossal-AI, you can find a simple example in the `feature` folder and a corresponding tutorial in feature section of the [**documentation**](https://www.colossalai.org/). For more complex examples for domain-specific models, you can find them in this repository as well. Some of them are covered in the advanced tutorials 
of the [**documentation**](https://www.colossalai.org/).

This repository is built upon Colossal-AI and Titans.

<div align="center">
    <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/repo_relation.png" width="300" title="repo-architecture">
</div>


### 🚀 Quick Links

[**Colossal-AI**](https://github.com/hpcaitech/ColossalAI) | 
[**Titans**](https://github.com/hpcaitech/Titans)
[**Paper**](https://arxiv.org/abs/2110.14883) | 
[**Documentation**](https://www.colossalai.org/) | 
[**Forum**](https://github.com/hpcaitech/ColossalAI/discussions) | 
[**Blog**](https://www.colossalai.org/) 

## Setup

1. Install Colossal-AI

You can download Colossal-AI [here](https://www.colossalai.org/download).

2. Install dependencies

```
pip install -r requirements.txt
```

## Table of Content

This repository contains examples of training models with ColossalAI. These examples fall under three categories:

1. Computer Vision
    - ResNet
    - SimCLR
    - Vision Transformer
        - Data Parallel
        - Pipeline Parallel
        - Hybrid Parallel
    - WideNet
        - Mixture of experts

2. Natural Language Processing
    - BERT
        - Sequence Parallel
    - GPT-2
        - Hybrid Parallel
    - GPT-3
        - Hybrid Parallel
    - Knowledge Graph Embedding

3. Features
    - Mixed Precision Training
    - Gradient Accumulation
    - Gradient Clipping
    - Tensor Parallel
    - Pipeline Parallel
    - ZeRO

The `image` and `language` folders are for complex model applications. The `features` folder is for demonstration of Colossal-AI. The `features` folder aims to be simple so that users can execute in minutes. Each example in the `features` folder relates to a tutorial in the [Official Documentation](https://colossalai.org/).

**If you wish to make contribution to this repository, please read the `Contributing` section below.**

## Discussion

Discussion about the Colossal-AI project and examples is always welcomed! We would love to exchange ideas with the community to better help this project grow.
If you think there is a need to discuss anything, you may jump to our [discussion forum](https://github.com/hpcaitech/ColossalAI/discussions) and create a topic there.

If you encounter any problem while running these examples, you may want to raise an issue in this repository.

## Contributing

This project welcomes constructive ideas and implementations from the community. 

### Update an Example

If you find that an example is broken (not working) or not user-friendly, you may put up a pull request to this repository and update this example.

### Add a New Example

If you wish to add an example for a specific application, please follow the steps below.

1. create a folder in the `image`, `language` or `features` folders. Generally we do not accept new examples for `features` as one example is often enough. **We encourage contribution with hybrid parallel or models of different domains (e.g. GAN, self-supervised, detection, video understanding, text classification, text generation)**
2. Prepare configuration files and `train.py`
3. Prepare a detailed readme on environment setup, dataset preparation, code execution, etc. in your example folder
4. Update the table of content (first section above) in this readme file


If your PR is accepted, we may invite you to put up a tutorial or blog in [ColossalAI Documentation](https://colossalai.org/).
