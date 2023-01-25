import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=True)
env.reset()

observation_space = env.observation_space
action_space = env.action_space

_ = env.reset()
action_mapper = {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left'
}

def run_episode(env, s0):
    _ = env.reset()
    step_counter = 0
    while True:
        action = np.random.randint(low=0, high=4)
        next_state, reward, done, info = env.step(action)
        step_counter += 1
        if done:
            break
    return step_counter

n_episode = 10
s0 = 6
for i in range(n_episode):
    len_ep = run_episode(env, s0)
    print('episode {} | numbers {}'.format(i, len_ep))