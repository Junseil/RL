import random
from collections import deque
import numpy as np


class SARSA_agent:
    def __init__(self,
                 ns: int,
                 na: int,
                 step: int,
                 epsilon: float
                 ):

        self.gamma = 0.99
        self.alpha = 0.05

        self.ns = ns
        self.na = na
        self.step = step
        self.epsilon = epsilon

        self.q = np.zeros((self.ns, self.na))

        self.states = deque()
        self.actions = deque()
        self.rewards = deque()

        self.GLIE = False

    def trajectory_update(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        if len(self.states) == self.step + 2:
            self.states.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.update()

    def trajectory_clear(self):
        if self.GLIE:
            self.epsilon *= 0.9
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()

    def update(self):
        s = self.states[0]
        a = self.actions[0]
        s_p = self.states[-1]
        a_p = self.actions[-1]

        self.q[s][a] -= self.alpha * self.q[s][a]
        sum_r = 0
        for r in reversed(self.rewards):
            sum_r *= self.gamma
            sum_r += r
        self.q[s][a]\
            += self.alpha * (sum_r + (self.gamma**self.step) * (self.q[s_p][a_p]-self.rewards[-1]))

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)

        if prob <= self.epsilon:
            action = np.random.choice(range(self.na))
        else:
            max_q = max(self.q[state])
            best_actions = [i for i, v in enumerate(self.q[state]) if v == max_q]
            action = random.choice(best_actions)
        return action
