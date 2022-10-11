# Example

Example of training resnet on cifar through different PP strategies.

## import data

```bash
export DATA=/path/cifar-10
```

## run Fill Drain
```bash
python3 fill_drain.py --epoch=1 --world_size=2 --batch_size=512 --chunk=1 --optimizer="SGD" --device="cuda" --num_microbatches=4
```

> for customized world_size, please adjust partition strategy


## run 1F1B

```bash
python3 1f1b.py --epoch=1 --world_size=2 --batch_size=512 --chunk=1 --optimizer="SGD" --device="cuda" --num_microbatches=4
```

> for customized world_size, please adjust partition strategy

## run Chimera
chimera is not stable, it is possible for the program here hang at some iteration.
```bash
python3 chimera.py --world_size=2 --epoch=1 --batch_size=128 --chun=1 --optimizer="SGD" --device="cuda" --num_microbatches=4
```

> for customized world_size, please adjust partition strategy

## help
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