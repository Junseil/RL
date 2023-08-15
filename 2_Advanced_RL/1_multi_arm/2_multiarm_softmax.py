import numpy as np
import random
import matplotlib.pyplot as plt


# 주어진 확률에 따른 0 ~ 10점 사이의 기대보상 획득
def get_reward(prob: float) -> int:
    reward = 0
    for i in range(10):
        if random.random() < prob:
            reward += 1
    return reward


# 새로운 action 과 reward를 획득했을시 기존의 간이 q-table update
def update_arm(arms: list, action: int, r: int) -> list:
    arms[action, 1] = (arms[action, 0] * arms[action, 1] + r) / (arms[action, 0] + 1)
    arms[action, 0] += 1
    return arms


# 각 확률을 지수를 통해 새롭게 조정 -> 기존 확률이 클수록 커지고, 작을수록 작아짐
# 의료, 자율주행등 오차가 심한 파장을 일으킬 경우 epsilon greedy 보다 선호됨
def softmax(x, tau):
    return np.exp(x / tau) / np.sum(np.exp(x / tau))


# subplot 생성
plt.subplots(1, 1)
plt.xlabel('plays')
plt.ylabel('avg reward')

# 필요변수들 선언
n = 10 # 슬롯머신의 머신 개수
arms = np.zeros((n, 2)) # 0열 - 당긴 횟수, 1열 - 갱신되는 평균
probs = np.random.rand(n) # 각 레버의 보상확률 초기화
epochs = 50000 # 레버를 돌릴 횟수
rewards = [0]

for i in range(epochs):
    # softmax 가 적용된 확률을 기반으로 랜덤하게 머신을 선택
    p = softmax(arms[:, 1], 1.12)
    choice = np.random.choice(np.arange(n), p=p)
    # 선택된 action 을 통해 q-table update
    r = get_reward(probs[choice])
    arms = update_arm(arms, choice, r)
    # 즉각적인 reward 들의 평균 추세
    rewards.append(((i+1)*rewards[-1] + r) / (i+2))

# 수렴하는지 확인
print(max(probs)*10)
print(rewards[-1])
# 즉각적인 reward 들의 평균 추세의 시각화
plt.scatter(np.arange(len(rewards)), rewards)
plt.show()
