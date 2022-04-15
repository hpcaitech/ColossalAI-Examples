# Knowledge Graph Entities Prediction by TransE

## Dataset
- Datasets in train.txt and dev.txt are triplets, [head_entity, relation, tail_entity]
- they are represented by unique numbers for convenience, which are mapped to actual words (entity or relation) in subset of FB15K-237
## Training & Evaluating
- TransE model. We use positive triplets from train.txt and negative triplets by random generation.
- Given the [head_entity, relation] in dev.txt, the trained model will find the most posible `n` tail_relations list. We check whether the actual tail_relation is in the list, and count to the global hit rate called `'hit@n'`
- run training by 
```
    python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500 transE_alt.py
```
It can run on a laptop with single NVIDIA GPU. Parralleled training is on testing.

## Baseline
- Batchsize, training rate and many other parameters can be changed. The default setting in the code may not be optimal.
- The best result I can get is:  
    `hit@1=0.162,hit@5=0.309`