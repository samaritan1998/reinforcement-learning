# Reinforcement Learning

本仓库包含强化学习算法的代码实现，对应于知乎强化学习专栏博客：[强化学习专栏](https://www.zhihu.com/column/c_1820091444070866944)。

## 目录

- [Reinforcement Learning](#reinforcement-learning)
  - [目录](#目录)
  - [简介](#简介)
  - [安装](#安装)
  - [算法实现](#算法实现)
    - [Q-Learning](#q-learning)
    - [Sarsa](#sarsa)
    - [DQN](#dqn)
    - [Policy Gradient](#policy-gradient)
    - [PPO](#ppo)
  - [环境](#环境)
  - [使用方法](#使用方法)
    - [运行测试](#运行测试)
    - [训练代理](#训练代理)
  - [项目结构](#项目结构)
  - [贡献指南](#贡献指南)
  - [许可证](#许可证)
  - [联系方式](#联系方式)

---

## 简介

该项目旨在提供清晰、整洁的强化学习算法实现，帮助学习者深入理解强化学习的核心概念和实践方法。代码包括常用的值函数方法和策略优化方法，并在经典的强化学习环境中进行测试。

---

## 安装

1. **克隆仓库**

   ```bash
   git clone https://github.com/yourusername/reinforcement-learning.git
   cd reinforcement-learning
   ```

2. **安装依赖项**

   使用以下命令安装所需的 Python 库：

   ```bash
   pip install -r requirements.txt
   ```

---

## 算法实现

### Q-Learning

基于离散化状态空间的 Q-Learning 算法，在 CartPole 环境中进行训练。

**运行 Q-Learning 测试**

```bash
python -m unittest tests/test_q_learning.py
```

**训练 Q-Learning Agent**

```bash
python algorithms/q_learning.py
```

### Sarsa

*待实现...*

### DQN

*待实现...*

### Policy Gradient

*待实现...*

### PPO

*待实现...*

---

## 环境

- **CartPole**：经典的倒立摆控制问题，位于 `environments/cartpole.py`。
- **GridWorld**：*待实现...*
- **MountainCar**：*待实现...*

---

## 使用方法

### 运行测试

执行以下命令，运行所有算法的单元测试：

```bash
python -m unittest discover tests
```

### 训练代理

以 Q-Learning 算法为例，训练代理：

```bash
python algorithms/q_learning.py
```

*请确保在 `main.py` 或相应的算法脚本中设置了训练参数和环境配置。*

---

## 项目结构

```
reinforcement-learning/
├── README.md
├── requirements.txt
├── environments/
│   ├── __init__.py
│   ├── cartpole.py
│   ├── grid_world.py
│   └── mountain_car.py
├── algorithms/
│   ├── __init__.py
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── dqn.py
│   ├── policy_gradient.py
│   └── ppo.py
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py
│   ├── neural_networks.py
│   ├── plotting.py
│   └── helpers.py
├── main.py
└── tests/
    ├── test_q_learning.py
    ├── test_sarsa.py
    ├── test_dqn.py
    ├── test_policy_gradient.py
    └── test_ppo.py
```

---

## 贡献指南

欢迎对本项目提出建议、报告问题或提交拉取请求。请在贡献之前阅读 [贡献指南](CONTRIBUTING.md)。

---

## 许可证

本项目采用 MIT 许可证，详细信息请参阅 [LICENSE](LICENSE) 文件。

---

## 联系方式

如果您有任何疑问或建议，欢迎通过知乎专栏与我联系：[强化学习专栏](https://www.zhihu.com/column/c_1820091444070866944)。

---

**备注**：本项目将持续更新，敬请期待更多强化学习算法的实现和优化！