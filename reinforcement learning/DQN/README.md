# colossalai-DQN
## overview
In colossalai, there are many ways to run DQN in a distributed way. However, for shallow networks like DQN,
using tensor-parallel, pipeline or ZeRO to reduce memory use of GPUs is unnecessary. Therefore, we only present a
naive DQN using data-parallel feature of colossalai. Readers could easily scale their training to multiple GPUs by
changing the number of GPUs used.

## Environments
In this example, we support `Cart_Pole` and `Atari games` such as `Pong`. We present two networks for different types
of environments. For `Atari games`, you should use networks with convolutional layers such as `Atari_Network` in this
example. For other simple `gym` environments, you could use similar network structure of `Cart_Pole_Network`.
### using other environments
If you want to use other environments in `gym` or `Atari games`, you could add your environment name in `Gym_envs` or
`Atari_envs` located in `./model.utils`. Note for `Atari` environment, we provide some wrappers to deal with original
environment input. Therefore, if you don't want to change these wrappers, you could simply use `Atari games` with
`NoFrameskip` such as `PongNoFrameskip-v4`. If you want to use customize your environment input, you should adapt
`init_atari_env`, `make_atari` in `./model.utils` and corresponding wrappers in `./model.wrappers` to support the
environment you are going to use.

## How to train DQN
After you install all packages, you could simply start your training on single node by using:

```Bash
torchrun --standalone --nproc_per_node=<num_gpus> main.py --config=./configs/<config_file> --from_torch
```

you need to replace `<num_gpus>` to number of GPUs used, and substitute `<config_file>` by config files you are
going to use. For customizing configuration files, you could check two config files for more information. We provide
detailed comments there. And we also provide logs and `tensorboardX` to record your DQN performance. you could make use
of them in your training.

## About data-parallel training
When you are using multiple GPUs for data-parallel training, every GPU would create their own replay buffer and
sample transitions from it using batch size in your configuration files. Every GPU would hold an DQN agent with the same
network structure which forwards with their own transitions data but backwards with global all-reduced gradient.
Therefore, keeping batch size same but using multiple GPUs for data-parallel to train this example is similar to
enlarge batch size. If you want to use the same batch size, please reduce your batch size to `batch_size / num_gpus`
in your configs. However, if you are going to use larger batch size, please scale your learning rate and reduce
the `total_step` in your configs for better performance.


## Other features of colossalai
As I mentioned before, using tensor-parallel, pipeline or ZeRO in training DQN is less effective. However, if you want to
explore other features usage, you could check [GPT example](https://github.com/hpcaitech/ColossalAI-Examples/blob/main/language/gpt/train_gpt.py#L52)
for `ZeRO` and `pipeline` usage. For using tensor-parallel, you could replace `torch.nn` layers of your networks with layers
in `colossalai.nn.layer.colossalai_layer`, and set your parallel setting in your configuration file. And colossalai
would automatically use tensor-parallel. More details please check [1D_tensor_parallel feature](https://www.colossalai.org/docs/features/1D_tensor_parallel).