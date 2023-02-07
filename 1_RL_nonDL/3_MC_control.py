# unknown model
# Policy iteration
# 1. PE : every-visit MC (Non incremental update)
# 2. PI : epsilon greedy (custom)

import gym
import numpy as np
import matplotlib.pyplot as plt
from src.MC_agent import MC_agent

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

ns = env.observation_space.n
na = env.action_space.n

epsilon = 0.999
agent = MC_agent(ns, na, epsilon)
num_epi = 3000

Q_norm = []
for _ in range(num_epi):
    s = 0
    env.reset()
    while True:
        action = agent.get_action(s)
        ns, reward, done, tmp, _ = env.step(action)
        agent.episode_stack(s, action, reward)
        s = ns
        if done:
            agent.update()
            Q_norm.append(np.linalg.norm(agent.q))
            if reward == 1:
                agent.episode_clear(GLIE=True)
            else:
                agent.episode_clear(GLIE=False)
            break

plt.plot(range(num_epi), Q_norm)
plt.xlabel('Num of episodes')
plt.ylabel('Q value')
plt.show()