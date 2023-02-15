import random
import numpy as np


class MC_agent:
    def __init__(self,
                 ns: int,
                 na: int,
                 epsilon: float):

        self.gamma = 0.99
        self.ns = ns
        self.na = na
        self.epsilon = epsilon

        self.n_q = np.zeros((self.ns, self.na))
        self.s_q = np.zeros((self.ns, self.na))
        self.q = np.zeros((self.ns, self.na))

        self.states = []
        self.actions = []
        self.rewards = []

        self.eps = 1e-10
        self.GLIE = False

    def episode_stack(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def episode_clear(self):
        if self.GLIE:
            self.epsilon *= 0.9
        self.states = []
        self.actions = []
        self.rewards = []

    def update(self):
        states = reversed(self.states)
        actions = reversed(self.actions)
        rewards = reversed(self.rewards)

        sum_r = 0
        for s, a, r in zip(states, actions, rewards):
            sum_r *= self.gamma
            sum_r += r

            self.n_q[s, a] += 1
            self.s_q[s, a] += sum_r

        self.compute_values()

    def compute_values(self):
        self.q = self.s_q / (self.n_q + self.eps)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)

        if prob <= self.epsilon:
            action = np.random.choice(range(self.na))
        else:
            max_q = max(self.q[state])
            tmp = [i for i, v in enumerate(self.q[state]) if v == max_q]
            action = random.choice(tmp)
        return action
