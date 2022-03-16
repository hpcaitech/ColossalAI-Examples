python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr 0.0.0.0 --master_port 29000 train.py --config=configs/bert_vanilla.py
