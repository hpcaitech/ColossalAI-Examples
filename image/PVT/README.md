## Colossal_PVT
Reproduce the PVT classification model with ColossalAI

## Background

This project is the reproduction of [PVT model](https://github.com/whai362/PVT) with [ColossalAI](https://github.com/hpcaitech/ColossalAI) tool.

## Envirionment setup
```
conda create -n coai python=3.6
conda activate coai
pip install -r requirements.txt
```

## Usage
### run train.py
```
python -m torch.distributed.launch --nproc_per_node=nproc_per_node
                                   --master_addr MASTER_ADDR
                                   --master_port MASTER_PORT
                                   --run_train.py
                                   --config=CONFIG_FILE
                                   --world_size=WORLD_SIZE
                                   --rank=RANK
                                   --local_rank=LOCAL_RANK
```
### run engine.py
```
python -m torch.distributed.launch --nproc_per_node=nproc_per_node
                                   --master_addr MASTER_ADDR
                                   --master_port MASTER_PORT
                                   --run_engine.py
                                   --config=CONFIG_FILE
                                   --world_size=WORLD_SIZE
                                   --rank=RANK
                                   --local_rank=LOCAL_RANK
```

## Cite us
```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```