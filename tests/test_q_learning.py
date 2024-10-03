# tests/test_q_learning.py
import unittest
from environments.cartpole import CartPoleEnv
from algorithms.q_learning import QLearningAgent

class TestQLearningAgent(unittest.TestCase):
    def test_training(self):
        env = CartPoleEnv()
        agent = QLearningAgent(env, num_episodes=10)  # 为了测试，减少训练轮数
        rewards = agent.train()
        self.assertEqual(len(rewards), agent.num_episodes)
        print("Training rewards:", rewards)

    def test_play(self):
        env = CartPoleEnv()
        agent = QLearningAgent(env, num_episodes=10)
        agent.train()
        agent.play(num_episodes=1)
        # 如果能运行到这里，说明 play 方法没有错误

if __name__ == '__main__':
    unittest.main()
