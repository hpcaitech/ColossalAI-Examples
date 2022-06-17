[WIP]

DATA=/mnt/huggingface/datasets/wikitext/wikitext-2/ TOKENIZER=/mnt/huggingface/tokenizers/bert/bert-base-uncased colossalai run --nproc_per_node=4 train.py --config=configs/bert_base_tp1d.py