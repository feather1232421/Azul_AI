# Azul-AI (花砖物语: 基于深度强化学习的桌游博弈项目)

本项目目标实现一个结合 **蒙特卡洛树搜索 (MCTS)** 与 **深度神经网络 (Policy-Value Network)** 的自博弈 AI，且AI能够自主学习并超越人类/启发式搜索水平。

## 🤝 团队协作 (Collaboration)
本项目由两人共同开发：
- **AI & Logic (本仓库)**: 
  - 负责核心游戏逻辑
  - AlphaZero 风格的 MCTS 搜索算法
  - 基于 PyTorch 实现的双头神经网络 (Policy Head / Value Head) 深度学习模型及训练。
- **UI & Client**: 负责游戏运行逻辑、表现层、用户交互及真人对战功能。
  - 开发者: [@PXbask](https://github.com/PXbask)
  - 客户端仓库: https://github.com/PXbask/AZUL

[comment]: <> (## 🎮 交互说明)

[comment]: <> (  - 目前仍未实现公开逻辑交互)

## 🚀 开发进度
- [x] **基础建设**
  - 完善游戏核心逻辑，做到“命令行小游戏”
  - 支持 get_legal_moves 进行动作合法性检查。
  - 实现 RandomAgent 随机决策与基于规则的 GreedyAgent 启发式搜索。
- [x] **强化学习** 
  - 使用 MaskablePPO 进行端到端的强化学习探索
  - 支持 Refined Mask 输出Action Mask
  - 最终PPO的强度胜率强于RandomAgent，但弱于GreedyAgent（与Greedy对战胜率只有10~15%）
- [x] **策略蒸馏 (Distillation)**
  - 采集 N-Step 搜索数据，通过行为克隆 (BC) 预训练神经网络。
  - 将 BC 权重注入神经网络架构。
  - 原始特征该为one-hot 编码，PPO胜率得到
- [ ] **强化学习终极目标 (MCTS + NN)**
  - [x] 实现 MCTS 树搜索框架，支持 Policy Prior 与 Value Evaluation。
  - [x] 训练目标重构：从 `(obs, action, z)` 转向 `(obs, pi, z)` 分布拟合。
  - [ ] 闭环 Self-play：通过 MCTS 迭代不断提升神经网络的先验能力。
  - [ ] 性能对标：目标是在同等算力下，通过 NN 剪枝达到 3-Step 启发式搜索的胜率水平。

## 📊 算法细节 (Algorithm Details)
- **State Representation (562-dim Tensor)**: 
  - 原始 142 维特征经过 **One-hot 编码** 预处理，最终为562维特征。
  - 包含：工厂状态（Factories）、中心区统计（Center）、玩家墙面（Wall）、模式行（Pattern Lines）、分数与地板惩罚。
  - 特点：消除颜色与位置的伪线性关系，增强模型对非线性决策空间的感知。

- **Training Objective**:
  - - **Policy Loss**: 使用 **Cross Entropy (交叉熵)** 拟合 MCTS 搜索产生的访问频率分布 $\pi$。
    - 公式: $L_{pi} = -\sum \pi \log(p)$
  - **Value Loss**: 使用 均方误差（MSE）拟合对局最终结果 $z$（$z \in [-1, 1]$）。