#!/usr/bin/env sh

## phase 1: self-supervised training
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 train_simclr.py

## phase 2: linear evaluation
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 train_linear.py