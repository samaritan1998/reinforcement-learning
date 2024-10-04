import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Policy Gradient 模型
class PolicyGradient(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=24):
        super(PolicyGradient, self).__init__()
        # 定义神经网络结构
        self.fc1 = nn.Linear(n_states, hidden_size)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_size, n_actions)  # 输出层

    def forward(self, x):
        # 前向传播，使用 ReLU 激活函数
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # 输出动作概率分布
        return x

# Policy Gradient 智能体实现
class PolicyGradientAgent:
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.99):
        # 初始化参数
        self.gamma = gamma  # 折扣因子
        self.lr = lr  # 学习率
        # 初始化策略网络
        self.model = PolicyGradient(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 存储回合数据
        self.rewards = []
        self.log_probs = []

    def choose_action(self, state):
        # 根据策略网络选择动作
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # 存储动作对数概率，以便后续计算梯度
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        # 存储每步的奖励
        self.rewards.append(reward)

    def learn(self):
        # 计算每一步的折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # 标准化奖励
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # 计算损失并反向传播
        loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            loss -= log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 清空回合数据
        self.rewards = []
        self.log_probs = []

    def train(self, env, episodes):
        # 训练智能体多个回合
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            while True:
                # 根据当前状态选择动作
                action = self.choose_action(state)
                # 执行动作并观察下一状态和奖励
                next_state, reward, done, _ = env.step(action)
                self.store_reward(reward)
                state = next_state
                total_reward += reward
                if done:
                    # 每回合结束后学习
                    self.learn()
                    print(f"Episode: {episode}, Total Reward: {total_reward}")
                    break