import numpy as np


class MDP:
    def __init__(self, gamma=0.99, error_tol=1e-3):
        self.gamma = gamma
        self.error_tol = error_tol

        self.env = None
        self.policy = None
        self.ns = None
        self.na = None
        self.P = None
        self.R = None

    def initialize_env(self, env):
        self.env = env
        self.ns = env.observation_space.n
        self.na = env.action_space.n
        self.policy = np.ones([self.ns, self.na]) / self.na

        self.P \
            = np.array([[[0 for _ in range(self.ns)] for _ in range(self.na)] for _ in range(self.ns)])
        self.R \
            = np.array([[0 for _ in range(self.na)] for _ in range(self.ns)])
        for i in range(self.ns):
            for j in range(self.na):
                [(p, s, r, t)] = self.env.P[i][j]
                self.P[i][j][s] = p
                self.R[i][j] = r

    def get_r_pi(self, policy):
        return (policy * self.R).sum(axis=-1)

    def get_p_pi(self, policy):
        return np.einsum('nam,na->nm', self.P, policy)

    def policy_evaluation(self, policy, v_0=None):
        r_pi = self.get_r_pi(policy)
        p_pi = self.get_p_pi(policy)

        if v_0 is None:
            old_v = np.zeros(self.ns)
        else:
            old_v = v_0
        while True:
            new_v = r_pi + self.gamma * np.matmul(p_pi, old_v)

            err = np.linalg.norm(new_v - old_v)
            if err < self.error_tol:
                break
            else:
                old_v = new_v
        return new_v

    def policy_improvement(self, policy, v_pi=None):
        if v_pi is None:
            v_pi = self.policy_evaluation(policy)
        q_pi = self.R + self.gamma * np.einsum('nam,m->na', self.P, v_pi)
        improved_policy = np.zeros_like(policy)
        improved_policy[np.arange(q_pi.shape[0]), q_pi.argmax(axis=1)] = 1
        return improved_policy

    def policy_iteration(self, policy_0):
        old_pi = policy_0
        old_v = np.zeros(self.ns)
        while True:
            new_v = self.policy_evaluation(old_pi, old_v)
            new_pi = self.policy_improvement(old_pi, new_v)

            err = np.linalg.norm(old_pi - new_pi)
            if err <= self.error_tol:
                break
            else:
                old_pi = new_pi
                old_v = new_v
        return new_pi

    def value_iteration(self):
        old_v = np.zeros(self.ns)
        while True:
            new_v \
                = (self.R + self.gamma * np.einsum('nam,m->na', self.P, old_v)).max(axis=-1)
            err = np.linalg.norm(new_v - old_v)
            if err <= self.error_tol:
                policy = self.policy_improvement(self.policy, new_v)
                break
            else:
                old_v = new_v
        return policy
