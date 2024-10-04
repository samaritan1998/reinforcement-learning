import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# DQN 智能体实现
class DQNAgent:
    def __init__(self, n_states, n_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64):
        # 初始化参数
        self.n_actions = n_actions
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 初始探索率
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减系数
        self.lr = lr  # 学习率
        self.batch_size = batch_size  # 经验回放的批量大小
        # 经验回放存储转换
        self.memory = deque(maxlen=2000)
        # 初始化 Q 网络
        self.model = DQN(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # 使用均方误差作为损失函数

    def choose_action(self, state):
        # 采用 epsilon-greedy 策略选择动作
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actions = self.model(state)
        return torch.argmax(actions).item()

    def store_transition(self, state, action, reward, next_state, done):
        # 存储转换到经验回放
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        # 只有当记忆库中有足够的样本时才进行学习
        if len(self.memory) < self.batch_size:
            return
        # 从经验回放中随机采样一个批量
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 将批量数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算当前 Q 值
        q_values = self.model(states).gather(1, actions).squeeze(1)
        # 计算下一状态的 Q 值
        next_q_values = self.model(next_states).max(1)[0].detach()
        # 计算目标 Q 值
        target = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算当前 Q 值与目标 Q 值之间的损失
        loss = self.criterion(q_values, target)
        # 反向传播损失并更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                # 存储转换到经验回放
                self.store_transition(state, action, reward, next_state, done)
                # 从经验中学习
                self.learn()
                state = next_state
                total_reward += reward
                if done:
                    print(f"Episode: {episode}, Total Reward: {total_reward}")
                    break
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
