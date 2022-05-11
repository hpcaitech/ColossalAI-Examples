# Train and test classic Knowledge Graph Embedding methods

Real-world knowledge bases are usually expressed as multi-relational graphs, which are collections of factual triplets, where each triplet represents 
a relation between a head entity and a tail entity. However, real-word knowledge bases are usually incomplete, which motivates the research of 
automatically predicting missing links. A popular approach for Knowledge Graph Completion (KGC) is to embed entities and relations into continuous 
vector or matrix space, and use a well-designed score function to measure the plausibility of the triplet (also known as Knowledge Graph Embedding).
In this example, we introduce three knowledge graph embedding methods DistMult, ComplEx and RotatE.

- DistMult Paper: [Embedding entities and relations for learning and inference in knowledge bases](https://arxiv.53yu.com/abs/1412.6575)
- ComplEx Paper: [Complex Embeddings for Simple Link Prediction](https://dl.acm.org/doi/abs/10.5555/3045390.3045609)
- RotatE Paper: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.53yu.com/abs/1902.10197)

## How to Prepare Datasets

```bash
# download data
cd knowledge_graph_embedding
git clone https://github.com/MiracleDesigner/data
```

Each dataset consists of five files: 
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

The data in the three files train.txt, valid.txt and test.txt are all triples (*entity1*, *relation*, *entity2*), and the data in entities.dict 
and relations.dict are the id corresponding to the entity and the relation, respectively. The data form is relatively simple, so the datasets do not require additional preprocessing.

## Implemented features

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling


## Run single-GPU training

For example, this command train a RotatE model on WIN18RR dataset for a single GPU.
```bash
colossalai run --nproc_per_node 1 train.py
 --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/wn18rr \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save results/RotatE_wn18rr_0 --test_batch_size 16 -de
```

## Run multi-GPU training
To run multi-GPU training on a single node, just change the `--nproc_per_node` parameter. For example, if `--nproc_per_node=4`, 4 GPUs on this machine will be
used for training. However, to make sure the model converges well, you should adjust your batch size and learning rate accordingly.
