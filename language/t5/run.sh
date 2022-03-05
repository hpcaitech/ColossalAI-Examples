#!/usr/bin/env sh
export DATA=./real_data/shuffled_deduplicated_urls.json

torchrun --standalone --nproc_per_node=1 train_t5.py --config=t5_configs/t5_vanilla.py --from_torch