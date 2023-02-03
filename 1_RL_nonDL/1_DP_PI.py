# FrozenLake-v1 / MDP policy iteration
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from src.mdp import mdp

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=False, render_mode='human')
env.reset()

agent = mdp()
agent.initialize_env(env)
policy = agent.policy_iteration(agent.policy)

s = 0
while True:
    env.render()
    ns, reward, done, tmp, info = env.step(policy[s].argmax(axis=0))
    s = ns
    if done:
        break