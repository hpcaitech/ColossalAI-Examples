## Train BERT with ZeRO

### About ZeRO

Zero redundancy optimizer is a memory-optimization method for large-scale model training. 
It shards tensors in optimizer states, gradients, and parameters so that large models can be accommodated by limited GPU memory.
Offloading techniques are integrated to further utilize the CPU memory space.
Colossal-AI has an optimized ZeRO module equipped with our unique chunk mechanism to maximize the memory utilization and higher training throughput.
More details can be found in our [documentation](https://www.colossalai.org/docs/features/zero_redundancy_and_zero_offload).

## Pretraining

details to be added.

## Fine-tuning

In this repository, we provided finetuning examples for different downstream tasks. Each section comes with step-by-step instructions to fine-tune the pretrained bert model.

### GLUE

1. Prepare the dataset

Execute the command below. This will create a `download` folder in the current directory. This folder contains the downstream task datasets.

```bash
bash ./scripts/download_finetune_dataset.sh
```

2. Fine-tuning

Run the fine-tuning script. This script by defualt uses 1 GPU only. If you wish to use more GPUs, you can change the batch size per GPU. 
The SOTA results are reproduced with global batch size 128.

```bash
bash ./scripts/run_finetune_glue.sh
```

Reproduced results:

| Metric | Value |
| -      | -     |
| F1     | 89.6  |
| Accurarcy | 85.3 |



