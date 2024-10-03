# algorithms/q_learning.py
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, buckets=(1, 1, 6, 12), num_episodes=500, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        """
        基于离散化状态空间的 Q-Learning 算法。

        参数：
        - env: 环境实例
        - buckets: 每个连续状态变量的离散桶数量
        - num_episodes: 训练的总轮数
        - alpha: 学习率
        - gamma: 折扣因子
        - epsilon: 初始探索率
        - epsilon_decay: 探索率的衰减因子
        """
        self.env = env
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(float)
        self.action_space = self.env.action_space.n

        # 环境状态的上下限
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        # 修正部分无穷大的状态边界
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-np.radians(50), np.radians(50)]

    def discretize(self, state):
        """将连续状态离散化为离散状态索引"""
        ratios = [(state[i] - self.state_bounds[i][0]) / (self.state_bounds[i][1] - self.state_bounds[i][0]) for i in range(len(state))]
        new_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))]
        new_state = [min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return tuple(new_state)

    def choose_action(self, state):
        """根据 ε-贪心策略选择动作"""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.Q[(state, a)] for a in range(self.action_space)]
            return int(np.argmax(q_values))

    def update_q(self, state, action, reward, next_state, done):
        """更新 Q 值"""
        best_next_action = np.argmax([self.Q[(next_state, a)] for a in range(self.action_space)])
        td_target = reward + self.gamma * self.Q[(next_state, best_next_action)] * (not done)
        td_error = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    def train(self):
        """训练代理"""
        rewards = []
        for episode in range(self.num_episodes):
            current_state = self.discretize(self.env.reset())
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, info = self.env.step(action)
                next_state = self.discretize(obs)
                self.update_q(current_state, action, reward, next_state, done)
                current_state = next_state
                total_reward += reward

            # 更新探索率
            self.epsilon *= self.epsilon_decay
            rewards.append(total_reward)
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        return rewards

    def play(self, num_episodes=5):
        """使用训练后的策略进行测试"""
        for episode in range(num_episodes):
            current_state = self.discretize(self.env.reset())
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                next_state = self.discretize(obs)
                current_state = next_state
                total_reward += reward

            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
        self.env.close()
