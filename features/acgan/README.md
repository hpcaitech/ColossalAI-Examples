# Auxiliary Classifier GAN Based on Colossal-AI

## Auxiliary Classifier Generative Adversarial Network

Synthesizing high resolution photorealistic images has been a long-standing challenge in machine learning. In this paper we introduce new methods for the improved training of generative adversarial networks (GANs) for image synthesis. We construct a variant of GANs employing label conditioning that results in 128x128 resolution image samples exhibiting global coherence. We expand on previous work for image quality assessment to provide two new analyses for assessing the discriminability and diversity of samples from class-conditional image synthesis models. These analyses demonstrate that high resolution samples provide class information not present in low resolution samples. Across 1000 ImageNet classes, 128x128 samples are more than twice as discriminable as artificially resized 32x32 samples. In addition, 84.7% of the classes have samples exhibiting diversity comparable to real ImageNet data.

You can access [here](https://arxiv.org/abs/1610.09585) to know more about AC-GAN.



## How to run 

```shell
python3 -m torch.distributed.launch --nproc_per_node <num_gpus> --master_addr localhost --master_port 29500 run_acgan_with_engine.py
```

Assume your pytorch version >= 1.10, you can directly run as

```shell
torchrun --standalone --nnodes=1 --nproc_per_node <num_gpus> run_acgan_with_engine.py
```
