# DEtection TRansformer (DETR) on Colossal-AI

## Requirement

You should install colossalai from the **latest** main branch.

---

## How to run

On a single server, you can directly use torch.distributed to start pre-training on multiple GPUs in parallel. In Colossal-AI, we provided several launch methods to init the distributed backend. You can use `colossalai.launch` and `colossalai.get_default_parser` to pass the parameters via command line. If you happen to use launchers such as SLURM, OpenMPI and PyTorch launch utility, you can use `colossalai.launch_from_<torch/slurm/openmpi>` to read rank and world size from the environment variables directly for convenience. 

Before running, you should `export DATA=/path/to/coco`.

In your terminal
```shell
colossalai run --nproc_per_node <world_size> main.py --config config.py
```

---


## Details
`config.py`

Containing configurations for DETR.

`main.py`

Engine is called through this file to start the training process using Colossal-AI.

`engine.py`

Process training and evaluating procedures about DETR.

`./datasets`

Dataset proprocessings.

`./models`

Model specifications of DETR model. Containing Transformer and Backbone implementations. 

`./util`

Utilities used in DETR.