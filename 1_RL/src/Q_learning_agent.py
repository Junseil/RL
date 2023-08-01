import random
import numpy as np


class Q_learning_agent:
    def __init__(self,
                 ns: int,
                 na: int,
                 epsilon: float
                 ):

        self.gamma = 0.99
        self.alpha = 0.05

        self.ns = ns
        self.na = na
        self.epsilon = epsilon

        self.q = np.zeros((self.ns, self.na))
        self.GLIE = False

    def update(self, s, a, r, ns):
        self.q[s][a] -= self.alpha * self.q[s][a]
        self.q[s][a] += self.alpha*(r + self.gamma * max(self.q[ns]))

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)

        if prob <= self.epsilon:
            action = np.random.choice(range(self.na))
        else:
            max_q = max(self.q[state])
            best_actions = [i for i, v in enumerate(self.q[state]) if v == max_q]
            action = random.choice(best_actions)
        return action

    def epsilon_decaying(self):
        if self.GLIE:
            self.epsilon *= 0.9

