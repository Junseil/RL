import gym

env = gym.make('Breakout-v4', render_mode='human')
env.reset()

for _ in range(100):
    env.reset()
    cnt = 0
    while True:
        ns, reward, done, tmp, info = env.step(env.action_space.sample())
        cnt += 1
        if done:
            break
