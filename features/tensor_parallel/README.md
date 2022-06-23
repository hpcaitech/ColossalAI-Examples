# Tensor Parallelism

## Usage

To use tensor parallelism, there are several steps to follow:

1. define `parallel` in your configuration file. Set `mode` for `tensor` to `1d`, `2d`, `2.5d` or `3d`.
2. construct your model, replace `torch.nn.Linear` with `colossalai.nn.Linear`.
3. split the input data accordingly

## Reference

If you wish to understand how tensor parallelism works exactly, you may refer to our [documentation](www.colossalai.org).


## How to run

In this example, we constructed a simple MLP model for demonstration purpose. You can execute the following commands to run the demo.

```shell
# run 1D tensor parallelism on 4 GPUs
colossalai run --nproc_per_node=4 run.py --config ./configs/tp_1d.py

# run 2D tensor parallelism 4 GPUs
colossalai run --nproc_per_node=4 run.py --config ./configs/tp_2d.py

# run 2.5D tensor parallelism 8 GPUs
colossalai run --nproc_per_node=8 run.py --config ./configs/tp_2p5d.py

# run 3D tensor parallelism 8 GPUs
colossalai run --nproc_per_node=8 run.py --config ./configs/tp_3d.py
```
