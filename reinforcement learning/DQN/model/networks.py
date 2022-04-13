import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import colossalai.nn as col_nn
from colossalai.utils import print_rank_0

class Cart_Pole_Network(nn.Module):
    """Networks for Cart Pole environment"""
    def __init__(self, input_shape, num_actions):
        super(Cart_Pole_Network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_shape[0], out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logit = self.out(x)
        return logit

class Atari_Network(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Atari_Network, self).__init__()

        self.input_shape = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=self._get_feature_size(), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, state):
        x = self.features(state)
        # flatten output of conv layers
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return logit

    def _get_feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)