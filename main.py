from environments.cartpole import CartPoleEnv
from algorithms.dqn import DQNAgent
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.policy_gradient import PolicyGradientAgent
from algorithms.ppo import PPOAgent


# 主函数，用于训练不同的智能体
def main():
    env = CartPoleEnv()  # 初始化环境
    episodes = 500  # 训练回合数

    # 训练 Q-Learning 智能体
    print("Training Q-Learning Agent...")
    q_learning_agent = QLearning(env.observation_space.n, env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.1)
    q_learning_agent.train(env, episodes)

    # 训练 SARSA 智能体
    print("Training SARSA Agent...")
    sarsa_agent = SARSA(env.observation_space.n, env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.1)
    sarsa_agent.train(env, episodes)


    # 训练 DQN 智能体
    print("Training DQN Agent...")
    dqn_agent = DQNAgent(n_states=env.observation_space.shape[0], n_actions=env.action_space.n)
    dqn_agent.train(env, episodes)

    # 训练 Policy Gradient 智能体
    print("Training Policy Gradient Agent...")
    agent = PolicyGradientAgent(n_states=env.observation_space.shape[0], n_actions=env.action_space.n)
    agent.train(env, episodes)

    # 训练 PPO 智能体
    print("Training PPO Agent...")
    agent = PPOAgent(n_states=env.observation_space.shape[0], n_actions=env.action_space.n)
    agent.train(env, episodes)

if __name__ == "__main__":
    main()