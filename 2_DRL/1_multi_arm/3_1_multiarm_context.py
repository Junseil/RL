import random
import numpy as np
import torch.nn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 훈련에 필요한 클래스 세팅
class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.bandit_matrix = np.random.rand(arms, arms)
        self.state = np.random.randint(0, self.arms)

    def reward(self, prob):
        r = 0
        for _ in range(self.arms):
            if random.random() < prob:
                r += 1
        return r

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.state][arm])

    def choose_arm(self, arm):
        r = self.get_reward(arm)
        self.state = arm
        return r


# 원핫벡터 생성기
def one_hot(n, pos):
    one_hot_vec = np.zeros(n)
    one_hot_vec[pos] = 1
    return one_hot_vec


# 훈련 준비
model = torch.nn.Sequential(
    torch.nn.Linear(10, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
    torch.nn.ReLU(),
)

loss_fn = torch.nn.MSELoss()
env = ContextBandit()

# 훈련 루프

cur_state = torch.Tensor(one_hot(10, env.state))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
rewards = []
part = 3
epochs = 50000

for i in range(epochs):
    y_pred = model(cur_state)
    av_softmax = np.exp(np.array(y_pred.data))
    av_softmax /= av_softmax.sum()
    action = np.random.choice(10, p=av_softmax)
    cur_reward = env.choose_arm(action)
    rewards.append(cur_reward)
    one_hot_reward = np.array(y_pred.data).copy()
    one_hot_reward[action] = cur_reward
    reward = torch.Tensor(one_hot_reward)
    loss = loss_fn(y_pred, reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cur_state = torch.Tensor(one_hot(10, env.state))

# plot 생성
N = int(epochs / 10)
rewards = np.array(rewards)
N_p = rewards.shape[0] - N
y = []
for i in range(N_p):
    y.append(sum(rewards[i:i+N]/N))
plt.plot(y)
plt.show()
