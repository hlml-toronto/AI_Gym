import gym


test_env = 'BipedalWalker-v3'  # switch this line to test different environments
env = gym.make(test_env)
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
