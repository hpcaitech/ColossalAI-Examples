# Bert

![Still In Progress](https://img.shields.io/badge/-Still%20In%20Progress-orange)

Bert Benchmark with data parallel, tensor parallel(tp), pipeline parallel(pp) and ZeRO.

## Setup
1. Install dependencies if you do not have them
```
pip install -r requirement.txt
```

2. Add root dir into PYTHONPATH
```
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
```

## Bert Usage

1. Prepare datasets and tokenizers from HuggingFace Hub if necessary (e.g. we provide an example of training `wikitext-2`).

2. Run benchmark with one of the systems to evaluate
```
DATA=/PATH/TO/DATASET TOKENIZER=/PATH/TO/TOKENIZER LOG=/PATH/TO/LOG torchrun --nproc_per_node=NUM_GPUS run.py --config=CONFIG_FILE
```