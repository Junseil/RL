import gym
import numpy as np
import torch

l1, l2, l3 = 4, 150, 2

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax()
)

def discount_rewards(rewards, gamma=0.99):
    len_r = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(len_r).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return

# 본가