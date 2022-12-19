# Example

Example of training OPT-125m through different PP strategies.

## run non-interleaved 1F1B

```bash
python3 1f1b.py --world_size=4 --num_microbatches=8 --device="cuda" --batch_size=16 --epoch=20 --master_port=29011
```

> for customized world_size, please adjust partition strategy