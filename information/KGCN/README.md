
# KGCN
- 2D tensor paralleled.
- if comment parallel settings in config.py, run on one GPU:   
```
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 run.py
```
- else uncomment parallel settings in config.py, run on four GPUs:  
```
python -m torch.distributed.launch --nproc_per_node 4 --master_addr localhost --master_port 29500 run.py
```
## Original Code From https://github.com/zzaebok/KGCN-pytorch 
- and below is the original readme
# KGCN-pytorch

This is the Pytorch implementation of [KGCN](https://dl.acm.org/citation.cfm?id=3313417) ([arXiv](https://arxiv.org/abs/1904.12575)):

> Knowledge Graph Convolutional Networks for Recommender Systems  
Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.  
In Proceedings of The 2019 Web Conference (WWW 2019)

## Dataset

- ### Movie

    Raw rating file for movie is too large to be contained in this repo.

    Downlad the rating data first
    ```bash
    $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
    $ unzip ml-20m.zip
    $ mv ml-20m/ratings.csv data/movie/
    ```

- ### Music

    Nothing to do

- ### Other dataset

    If you want to use your own dataset, you need to prepare 2 data.

    1. Rating data
        - Each row should contain (user-item-rating)
        - In this repo, it is pandas dataframe structure. (look at `data_loader.py`)
    2. Knowledge graph
        - Each triple(head-relation-tail) consists of knowledge graph
        - In this repo, it is dictionary type. (look at `data_loader.py`)

## Structure
1.  `data_loader.py`
    - data loader class for movie / music dataset
    - you don't need it if you make custom dataset

2. `aggregator.py`
    - aggregator class which implements 3 aggregation functions

3. `model.py`
    - KGCN model network

## Running the code

Look at the `KGCN.ipynb`.

It contains
- how to construct Datset
- how to construct Data loader
- how to train network