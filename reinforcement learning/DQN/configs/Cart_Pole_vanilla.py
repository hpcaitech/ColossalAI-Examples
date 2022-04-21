from model.networks import Cart_Pole_Network
from torch.optim import Adam
from torch.nn import MSELoss
from colossalai.nn.lr_scheduler import MultiStepLR

# parameters for colossalai DQN should conclude three parts

# 1. params for DQN learning
environment = "CartPole-v1"
total_step = 10000
batch_size = 32
learning_rate = 0.01
gamma = 0.9
# capacity of replay buffer
memory_capacity = 5000
# number of steps to store transitions before DQN learning
pre_step = 800
# maximum steps for one episode
max_steps = 400
# steps to update target net
update_iteration = 100
# param for getting epsilon
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


# 2. params for models
model = dict(
    type=Cart_Pole_Network,
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
    milestones=[40, 80],
    gamma=0.1
)

# 3. params for colossalai features (optional)
# for vanilla config, there would be None
