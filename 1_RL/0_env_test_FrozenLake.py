import gym

env = gym.make('FrozenLake-v1', map_name='8x8', render_mode='human', is_slippery=False)
env.reset()
num_s = env.observation_space.n
num_a = env.action_space.n
P = env.P # P(s, a) -> return p, s, r, t

action_mapper = {
    0: 'left',
    1: 'down',
    2: 'right',
    3: 'up'
}

# cnt = 0
# while True:
#     print(f'At t = {cnt+1}')
#     env.render()
#
#     cur_state = env.s
#     action = env.action_space.sample()
#     ns, reward, done, tmp, info = env.step(action)
#
#     print(f'state = {cur_state}')
#     print(f'action = {action_mapper[action]}')
#     print(f'next state = {ns}')
#     print(f'reward = {reward}')
#     print()
#     cnt += 1
#     if done:
#         break

def run_episode(env):
    env.reset()
    cnt = 0
    while True:
        ns, reward, done, tmp, info = env.step(env.action_space.sample())
        cnt += 1
        if done:
            break
    return cnt

for i in range(10):
    print(f'Episode {i+1} | length of episode : {run_episode(env)}')