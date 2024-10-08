import numpy as np

# Q-Learning 算法实现
class QLearning:
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

    def learn(self, state, action, reward, next_state):
        # 使用 Q-Learning 更新规则更新 Q 值
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, episodes):
        # 训练智能体多个回合
        for episode in range(episodes):
            state = env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                if done:
                    break