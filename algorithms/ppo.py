import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# PPO 模型
class PPO(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=64):
        super(PPO, self).__init__()
        # 定义策略网络
        self.fc1 = nn.Linear(n_states, hidden_size)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接层
        self.action_head = nn.Linear(hidden_size, n_actions)  # 动作输出层
        self.value_head = nn.Linear(hidden_size, 1)  # 状态值输出层

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)  # 输出动作概率分布
        state_value = self.value_head(x)  # 输出状态值
        return action_probs, state_value

# PPO 智能体实现
class PPOAgent:
    def __init__(self, n_states, n_actions, lr=0.001, gamma=0.99, clip_epsilon=0.2, k_epochs=4):
        # 初始化参数
        self.gamma = gamma  # 折扣因子
        self.clip_epsilon = clip_epsilon  # 剪切范围
        self.k_epochs = k_epochs  # 训练次数
        # 初始化策略网络和优化器
        self.model = PPO(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 存储回合数据
        self.memory = []

    def choose_action(self, state):
        # 根据策略网络选择动作
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def store_transition(self, transition):
        # 存储状态、动作、奖励等信息
        self.memory.append(transition)

    def learn(self):
        # 从存储的回合数据中提取信息
        states, actions, log_probs_old, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        log_probs_old = torch.stack(log_probs_old)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算每一步的折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative_reward = 0
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # 标准化奖励
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 进行多次更新
        for _ in range(self.k_epochs):
            action_probs, state_values = self.model(states)
            action_dist = Categorical(action_probs)
            log_probs_new = action_dist.log_prob(actions.squeeze())
            # 计算比率
            ratios = torch.exp(log_probs_new - log_probs_old)
            # 计算策略损失
            advantages = discounted_rewards - state_values.squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # 计算值函数损失
            value_loss = nn.MSELoss()(state_values.squeeze(), discounted_rewards)
            # 总损失
            loss = policy_loss + 0.5 * value_loss
            # 反向传播并更新网络参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空回合数据
        self.memory = []

    def train(self, env, episodes):
        # 训练智能体多个回合
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            while True:
                # 根据当前状态选择动作
                action, log_prob = self.choose_action(state)
                # 执行动作并观察下一状态和奖励
                next_state, reward, done, _ = env.step(action)
                # 存储转换到内存中
                self.store_transition((state, action, log_prob, reward, next_state, done))
                state = next_state
                total_reward += reward
                if done:
                    # 每回合结束后学习
                    self.learn()
                    print(f"Episode: {episode}, Total Reward: {total_reward}")
                    break
