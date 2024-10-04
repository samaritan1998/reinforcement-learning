# 强化学习代码库

本仓库包含强化学习算法的代码实现，对应于知乎强化学习专栏博客：[强化学习专栏](https://www.zhihu.com/column/c_1820091444070866944), 包含以下强化学习算法的实现：
- Q-Learning
- SARSA
- 深度 Q 网络 (DQN)
- Policy Gradient
- Proximal Policy Optimization (PPO)

## 文件结构

```
reinforcement-learning/
├── environments/
│   ├── cartpole.py
├── algorithms/
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── dqn.py
│   ├── policy_gradient.py
│   └── ppo.py
├── main.py
└── README.md
```

- `environments/cartpole.py`：包含自定义的环境类，用于包装 OpenAI Gym 环境。
- `algorithms/q_learning.py`：Q-Learning 算法的实现。
- `algorithms/sarsa.py`：SARSA 算法的实现。
- `algorithms/dqn.py`：深度 Q 网络 (DQN) 的实现。
- `algorithms/policy_gradient.py`：Policy Gradient 算法的实现。
- `algorithms/ppo.py`：Proximal Policy Optimization (PPO) 算法的实现。
- `main.py`：用于训练不同智能体的脚本。
- `README.md`：项目说明文件。

## 环境设置

### 依赖项

运行本项目需要以下依赖项：

- Python 3.7+
- Gym
- Numpy
- PyTorch

可以使用以下命令安装依赖项：

```sh
pip install -r requirements.txt
```

`requirements.txt` 文件内容如下：

```
numpy
gym
torch
```

## 使用说明

可以使用 `main.py` 文件来训练不同的智能体。

```sh
python main.py
```

`main.py` 文件中包含训练以下智能体的逻辑：

- SARSA
- Q-Learning
- DQN
- Policy Gradient
- PPO

每个智能体将在 OpenAI Gym 的 `CartPole-v1` 环境中进行训练。

## 联系方式

如果您对代码有任何问题或建议，欢迎提交 issue 或联系我。

