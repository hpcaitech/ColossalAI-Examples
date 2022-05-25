# colossal_detr
Reproduce the DETR model with ColossalAI

## Background
This project is the reproduction of [DETR model](https://arxiv.org/abs/2005.12872) with [ColossalAI](https://github.com/hpcaitech/ColossalAI) tool.

## Environment setup
```
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

## How to run
```
$ DATA=/path/to/data/ python -m torch.distributed.launch --nproc_per_node=nproc_per_node
                                                         --master_addr MASTER_ADDR
                                                         --master_port MASTER_PORT
                                                         run_train.py
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