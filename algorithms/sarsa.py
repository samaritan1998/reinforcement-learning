import numpy as np


# SARSA 算法实现
class SARSA:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        # 初始化参数
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        # 初始化 Q 表，所有值为 0
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # 采用 epsilon-greedy 策略选择动作
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, next_action):
        # 使用 SARSA 更新规则更新 Q 值
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, episodes):
        # 训练智能体多个回合
        for episode in range(episodes):
            state = env.reset()
            action = self.choose_action(state)
            while True:
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(next_state)
                self.learn(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                if done:
                    break