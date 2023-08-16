import sys
sys.path.append('C:/Users/ypd07/PycharmProjects/study/RL/2_Advanced_RL')
for i in sys.path:
    print(i)

import gym
import numpy as np
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from src.MLP import MLP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('Breakout-v4', render_mode='human')
env.reset()

for _ in range(100):
    env.reset()
    cnt = 0
    while True:
        ns, reward, done, truncation, info = env.step(env.action_space.sample())
        cnt += 1
        if done:
            break
