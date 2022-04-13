import math
import random
import numpy as np
import gym
from collections import deque
from gym import wrappers

from .wrappers import *


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def get_epsilon_by_step(step, epsilon_start, epsilon_final, epsilon_decay):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step / epsilon_decay)


# utils for initializing atari environment
def init_atari_env(env_id):
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


# Add your Atari-env here
Atari_envs = ["PongNoFrameskip-v4", "BoxingNoFrameskip-v4"]

# Add your gym-env here
Gym_envs = ["CartPole-v1",]