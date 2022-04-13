from model.networks import Atari_Network
from torch.optim import Adam
from torch.nn import MSELoss
from colossalai.nn.lr_scheduler import MultiStepLR

# parameters for colossalai DQN should conclude three parts

# 1. params for DQN learning
environment = "PongNoFrameskip-v4"
total_step = 1000000
batch_size = 32
learning_rate = 0.0001
gamma = 0.9
# capacity of replay buffer
memory_capacity = 100000
# number of steps to store transitions before DQN learning
pre_step = 800
# maximum steps for one episode
max_steps = 10000
# steps to update target net
update_iteration = 100
# param for getting epsilon
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


# 2. params for models
model = dict(
    type=Atari_Network,
)

optimizer = dict(
    type=Adam,
    lr=learning_rate,
)

loss = dict(
    type=MSELoss,
)

lr_shceduler = dict(
    type=MultiStepLR,
    total_steps=total_step,
    milestones=[100, 300, 500],
    gamma=0.1
)

# 3. params for colossalai features (optional)
# for vanilla config, there would be None
