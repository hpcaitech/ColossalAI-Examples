import torch

state_dict = torch.load('vit_cifar.pt')
print(state_dict['model'].keys())