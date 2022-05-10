# GPT2 ZeRO Benchmark
GPT2 ZeRO benchmark with data parallelism to evaluate Colossal-AI, DeepSpeed, FairScale and PatrickStar.

## Requirements
```
CUDA>=11.3
torch>=1.10.0
deepspeed>=0.5.8
fairscale>=0.4.5
patrickstar>=0.4.6
nvidia-dali>=1.8.0
```

## Setup
1. Install dependencies if you do not have them
```
pip install -r requirement.txt
```
2. Also, download PatrickStar from github
```
https://github.com/Tencent/PatrickStar.git
```
3. Install PatrickStar
```
cd PatrickStar
pip install .
```
4. Add root dir into PYTHONPATH
```
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
```

## GPT Usage

1. Prepare datasets and tokenizers from HuggingFace Hub if necessary (e.g. we provide an example of training `wikitext-2`).

2. Run benchmark with one of the systems to evaluate
```
DATA=/PATH/TO/DATASET TOKENIZER=/PATH/TO/TOKENIZER LOG=/PATH/TO/LOG torchrun --nproc_per_node=NUM_GPUS run.py --config=CONFIG_FILE
```

## VIT Usage
1. Prepare ImageNet-1k datasets (TFrecord version).

2. Run benchmark with one of the systems to evaluate
```
DATA=/PATH/TO/DATASET LOG=/PATH/TO/LOG torchrun --nproc_per_node=NUM_GPUS run.py --config=CONFIG_FILE
```
