import numpy as np

class MCagent:
    def __init__(self,
                 gamma: float,
                 ns: int,
                 na: int,
                 epsilon: float):
        self.gamma = gamma
        self.ns = ns
        self.na = na
        self.epsilon = epsilon

        self.eps = 1e-10

        self.n_v = np.zeros(self.ns)
        self.s_v = np.zeros(self.ns)
        self.n_q = np.zeros(self.ns, self.na)
        self.s_q = np.zeros(self.ns, self.na)

    def update(self, episode):
        states, actions, rewards = episode
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        sum_r = 0
        for s, a, r in zip(states, actions, rewards):
            sum_r *= self.gamma
            sum_r += r

            self.n_v[s] += 1
            self.n_q[s, a] += 1

            self.s_v[s] += sum_r
            self.s_q[s, a] += sum_r