# no CNN
# Receive physical information of the cart instead of image input.
# On-policy
# Does not use replay memory.
# A simple implementation of epsilon decaying.

import gym
import numpy as np
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from src.MLP import MLP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = gym.make('CartPole-v1')
env.reset()
s_dim = env.observation_space.shape[0]
'''
observation_space
0 - cart position
1 - cart velocity
2 - pole angle
3 - pole angular velocity
'''
a_dim = env.action_space.n


class Naive_DQN(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 q_net: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        super(Naive_DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = q_net
        self.lr = lr
        self.gamma = gamma

        self.opt = torch.optim.Adam(params=self.q_net.parameters(), lr=lr)
        self.criteria = nn.MSELoss()

        self.register_buffer('epsilon', torch.ones(1) * epsilon)

    def get_action(self, state):
        qs = self.q_net(state)

        if self.train:
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(range(self.action_dim))
            else:
                action = qs.argmax(dim=-1)
        else:
            action = qs.argmax(dim=-1)
        return int(action)

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        with torch.no_grad():
            q_max, _ = self.q_net(next_state).max(dim=-1)
            q_target = r + self.gamma * q_max * (1 - done)

        loss = self.criteria(self.q_net(s)[0, a], q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class Moving_average:
    def __init__(self,
                 alpha: float):
        self.s = None
        self.alpha = alpha

    def update(self, y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1 - self.alpha) * self.s


if __name__ == '__main__':
    qnet = MLP(input_dim=s_dim,
               output_dim=a_dim,
               num_neurons=[16, 32, 32],
               hidden_act='ReLU',
               out_act='Identity')

    agent = Naive_DQN(state_dim=s_dim,
                      action_dim=a_dim,
                      q_net=qnet,
                      lr=1e-4,
                      gamma=0.99,
                      epsilon=1.0)

    # training loop
    n_eps = 10000
    print_every = 200
    alpha = 0.1
    ma = Moving_average(alpha)
    y = []

    for ep in range(n_eps):
        env.reset()
        cum_r = 0
        while True:
            s = env.state
            s = torch.tensor(s).float().view(1, 4)
            a = agent.get_action(s)
            ns, r, done, truncation, info = env.step(a)

            ns = torch.tensor(ns).float()
            agent.update_sample(s, a, r, ns, done)
            cum_r += r
            if done or truncation:
                ma.update(cum_r)
                y.append(ma.s)
                if ep % print_every == 0:
                    print(f"Episode {ep} || MA: {ma.s:.1f} || EPS : {agent.epsilon}")
                if ep >= 2000:
                    agent.epsilon *= 0.999
                break
    env.close()

    y2 = np.zeros(len(y))
    for i in range(len(y)):
        if i >= 99:
            y2[i] = np.average(y[i-99:i+1])
    plt.plot(y2)
    plt.show()

    torch.save(agent.state_dict(), '1_Naive_DQN.pt')
