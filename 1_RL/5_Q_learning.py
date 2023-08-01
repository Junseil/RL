# unknown model
# Policy iteration
# 1. PE : every-visit MC (Non incremental update)
# 2. PI : epsilon greedy (custom)
# 2-1. When selecting an action with Greedy,
#       if there are multiple actions with the maximum Q value, select one of them at random.
# 2-2. The GLIE condition triggers from the moment agent first get the reward.
# Off-policy

import gym
import numpy as np
import matplotlib.pyplot as plt
from src.Q_learning_agent import Q_learning_agent

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

ns = env.observation_space.n
na = env.action_space.n

epsilon = 0.999
agent = Q_learning_agent(ns, na, epsilon)
num_epi = 500

Q_norm = []
for _ in range(num_epi):
    s = 0
    env.reset()
    agent.epsilon_decaying()
    while True:
        action = agent.get_action(s)
        ns, reward, done, tmp, _ = env.step(action)
        agent.update(s, action, reward, ns)
        s = ns
        if done:
            if reward != 0 and not agent.GLIE:
                agent.GLIE = True
            Q_norm.append(np.linalg.norm(agent.q))
            break

plt.plot(range(num_epi), Q_norm)
plt.xlabel('Num of episodes')
plt.ylabel('l2 norm of Q')
plt.show()
