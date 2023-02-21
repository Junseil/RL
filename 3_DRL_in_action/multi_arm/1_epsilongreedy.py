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


# greedy 방법을 통한 최대의 기대보상을 갖는 action (0~9 슬롯) 을 return
def get_best_arm(arms: list) -> int:
    return np.argmax(arms[:, 1], axis=0)


# subplot 생성
plt.subplots(1, 1)
plt.xlabel('plays')
plt.ylabel('avg reward')

# 필요변수들 선언
n = 10 # 슬롯머신의 머신 개수
arms = np.zeros((n, 2)) # 0열 - 당긴 횟수, 1열 - 갱신되는 평균
probs = np.random.rand(n) # 각 레버의 보상확률 초기화
eps = 0.001 # 최소의 탐험율 계수 eps
epochs = 50000 # 레버를 돌릴 횟수
rewards = [0]

for i in range(epochs):
    # epsilon greedy 를 통한 action 선택
    # epsilon 을 점차적으로 줄여나감 / exploration, exploitation 비중 변화
    if random.random() > max(eps, (epochs-epochs*(1/100)*i) / epochs):
        choice = get_best_arm(arms)
    else:
        choice = np.random.randint(n)
    # 선택된 action 을 통해 q-table update
    r = get_reward(probs[choice])
    arms = update_arm(arms, choice, r)
    # 즉각적인 reward 들의 평균 추세
    rewards.append(((i+1)*rewards[-1] + r) / (i+2))

# 수렴하는지 확인
print(max(probs)*10*(1-eps))
print(rewards[-1])
# 즉각적인 reward 들의 평균 추세의 시각화
plt.scatter(np.arange(len(rewards)), rewards)
plt.show()
