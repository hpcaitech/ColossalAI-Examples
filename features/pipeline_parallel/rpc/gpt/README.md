# Example

Example of training gpt-2 on gpt_data through different PP strategies.

## import data

```bash
export DATA=/path/small-gpt-dataset.json
```


## run non-interleaved 1F1B

```bash
python3 1f1b.py --world_size=4 --num_microbatches=8 --device="cuda" --batch_size=16 --epoch=20 --master_port=29011
```

> for customized world_size, please adjust partition strategy

## run interleaved 1F1B

```bash
python3 1f1b.py --world_size=2 --chunk=2 --num_microbatches=8 --device="cuda" --batch_size=16 --epoch=20 --master_port=29011
```

> for customized world_size, please adjust partition strategy

<<<<<<< HEAD
## run baseline (pippy)

Install Pippy first: https://github.com/pytorch/tau

Just copy the folder `pippy` and move to the site-packages of your current *Python Interpreter*.

Baseline (pippy) is not stable for the problem of communication. Rerun the program to acquire value of performance :D

```bash
python3 baseline.py --world_size=5 --schedule="1F1B" --batch_size=16 --chunk=8
```

> pippy need an extra rank for controlling, so if your original configure is "world_size==4", use "world_size=5" in pippy

## help 
=======

## help
>>>>>>> e1be6c9181ae85eca5bea1861214a93f192bbf53
run `python3 1f1b.py --help` for available config of the pipeline:

```
  -h, --help            show this help message and exit
  --epoch EPOCH
  --world_size WORLD_SIZE
  --batch_size BATCH_SIZE
  --dp_degree DP_DEGREE
  --tp_degree TP_DEGREE
  --num_microbatches NUM_MICROBATCHES
  --chunk CHUNK
  --use_checkpoint
  --optimizer {SGD,Adam,RMSprop}
  --device {cpu,cuda}
  --master_addr MASTER_ADDR
  --master_port MASTER_PORT
  --num_worker_threads NUM_WORKER_THREADS
```

`chunk` means the number of the virtual pipeline stages on each card. If `chunk==1`, then there is only one virtual stage on each card, equivalent to **non-overleaved** mode.

If `chunk>1`(`chunk=2` for example) then there are two virtual stages on each card, equivalent to **overleaved** mode.

As a result, actual number of pipeline stage (donated to `actual_stage_num`) is $\text{chunk} \times \text{world\_size}$.

It is recommended not to set `chunk>2`, too much communication payload on one card may make `torch.distributed.rpc` go wrong. It depends on your hardware.

In the demo of resnet, please set `worlds_size=2, chunk=1`, because current partition strategy only support this config.


## About partition

If you want to customize training schema, it is necessary to write `data_process_func` to fit the partition result.

There is the demo of `data_process_func`.

### world_size=2, balanced partition

```python
def data_process_func(pp_rank: int, args_kwargs):
    if pp_rank == 0:
        args = args_kwargs[0]
        kwargs = args_kwargs[1]
        return args, kwargs

    if pp_rank == 1:
        attention = args_kwargs[1]
        x = args_kwargs[0]
        args = (x, )
        kwargs = {"attention_mask": attention}
        return args, kwargs
```


### world_size=4, balanced partition

```python
def data_process_func(pp_rank: int, args_kwargs):
    if pp_rank == 0:
        args = args_kwargs[0]
        kwargs = args_kwargs[1]
        return args, kwargs

    elif pp_rank == 1:
        args = [args_kwargs[0]]
        kwargs = {"attention_mask" : None}
        return args, kwargs
    
    elif pp_rank == 2:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs

    elif pp_rank == 3:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs
```

### world_size=4, uniform partition

```python
def data_process_func(pp_rank: int, args_kwargs):
    if pp_rank == 0:
        args = args_kwargs[0]
        kwargs = args_kwargs[1]
        return args, kwargs

    elif pp_rank == 1:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs
    
    elif pp_rank == 2:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs

    elif pp_rank == 3:
        x = args_kwargs[0]
        attention_mask = args_kwargs[1]
        args = [x]
        kwargs = {"attention_mask" : attention_mask}
        return args, kwargs
```