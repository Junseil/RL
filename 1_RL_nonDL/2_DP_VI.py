# FrozenLake-v1 / MDP value iteration
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from src.mdp import MDP

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=False, render_mode='human')
env.reset()

agent = MDP()
agent.initialize_env(env)
policy = agent.value_iteration()

s = 0
while True:
    env.render()
    ns, reward, done, tmp, _ = env.step(policy[s].argmax(axis=0))
    s = ns
    if done:
        break