# environments/cartpole.py
import gym

class CartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()
