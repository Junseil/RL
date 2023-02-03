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

    def update(self,):