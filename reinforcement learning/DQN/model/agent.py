import numpy as np
import torch
from abc import ABC, abstractmethod

from .utils import ReplayBuffer


class base_DQN(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def choose_action(self, state, epsilon):
        pass

    @abstractmethod
    def store_transition(self, transition):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def step_learn(self):
        pass

class naive_DQN(base_DQN):
    def __init__(self, eval_engine, target_engine, batch_size, num_actions, memory_capacity, gamma, normalization=False):
        super(naive_DQN, self).__init__()

        # build networks for target_net and colossalai engine for eval_net
        self.eval_net, self.target_net = eval_engine, target_engine

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.replay_buffer = ReplayBuffer(memory_capacity)
        self.gamma = gamma
        # whether to normalize reward
        self.normalization = normalization

    def choose_action(self, state, epsilon):
        # get a 1D array
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()

        # epsilon greedy policy
        if np.random.rand(1) > epsilon:
            action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].item()
        # random policy
        else:
            action = np.random.choice(range(self.num_actions), 1).item()
        return action

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        self.target_net.model.load_state_dict(self.eval_net.model.state_dict())

    def step_learn(self):
        # sample batch transitions
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(self.batch_size)

        # turn into tensor and move to gpu
        batch_state = torch.FloatTensor(batch_state).cuda()  # size: [batch_size, num_state]
        # batch_action should be int64 for gather function, size: [batch_size, 1]
        batch_action = torch.Tensor(batch_action).type(torch.int64).view(self.batch_size, 1).cuda()
        batch_reward = torch.FloatTensor(batch_reward).cuda()  # size: [batch_size]
        batch_done = torch.Tensor(batch_done).type(torch.int64).cuda()  # size: [batch_size]
        batch_next_state = torch.FloatTensor(batch_next_state).cuda()  # size: [batch_size, num_state]

        # reward normalization
        if self.normalization:
            batch_reward = (batch_reward - batch_reward.mean()) / (batch_reward.std() + 1e-7)
        # get target Q-value
        with torch.no_grad():
            Q_target = batch_reward + self.gamma * self.target_net(batch_next_state).max(1)[0] * (1 - batch_done)

        # move tensor to gpu
        target_Q_value = Q_target.contiguous().cuda()  # size: [batch_size]

        # one batch step
        self.eval_net.zero_grad()
        Q_value = self.eval_net(batch_state).gather(1, batch_action)  # size: [batch_size, 1]
        loss = self.eval_net.criterion(Q_value, target_Q_value.unsqueeze(1))
        self.eval_net.backward(loss)
        self.eval_net.step()
        return loss
