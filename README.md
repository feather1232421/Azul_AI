# Azul-AI (花砖物语: 基于深度强化学习的桌游博弈项目)

本项目目标实现一个结合 **蒙特卡洛树搜索 (MCTS)** 与 **深度神经网络 (Policy-Value Network)** 的自博弈 AI，且AI能够自主学习并超越人类/启发式搜索水平。

## 📌 当前状态（2026-06）
- 当前主线为 **Transformer Policy-Value Network + MCTS**。
- `server.py` 默认加载 `models/transformer_champion.pt`，当前 champion 来源于 `transformer_candidate_20260519_213341.pt`。
- 已建立人工审核的 curated cases，用于回归检查关键妙手/臭手局面；当前 `check_curated_cases.py` 结果为 `2/3`。
- MCTS 小分支预算已针对实战配置做过一轮调整：当合法步较少时，会给出更高搜索预算，避免残局被过度截断。
- 当前建议：继续沿着 **self-play -> 训练新模型 -> arena 对比 -> 真人对战补 case** 的闭环迭代，而不是回退到 PPO 路线。

## 🤝 团队协作 (Collaboration)
本项目由两人共同开发：
- **AI & Logic (本仓库)**: 
  - 负责核心游戏逻辑
  - AlphaZero 风格的 MCTS 搜索算法
  - 基于 PyTorch 实现的双头神经网络 (Policy Head / Value Head) 深度学习模型及训练。
- **UI & Client**: 负责游戏运行逻辑、表现层、用户交互及真人对战功能。
  - 开发者: [@PXbask](https://github.com/PXbask)
  - 客户端仓库: https://github.com/PXbask/AZUL
  
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
- [ ] **强化学习终极目标 (MCTS + NN / Transformer)**
  - [x] 实现 MCTS 树搜索框架，支持 Policy Prior 与 Value Evaluation。
  - [x] 训练目标重构：从 `(obs, action, z)` 转向 `(obs, pi, z)` 分布拟合。
  - [x] 修复 MCTS 选择阶段的视角一致性问题（`best_child` 中的 exploitation 项）。
  - [x] 建立 Greedy 教师数据采集脚本，可用于冷启动训练。
  - [x] 闭环 Self-play：当前已具备 `run_iteration.py` / `loop_train.py` 驱动的 transformer 自博弈训练主线。
  - [ ] curated dataset 正式接入训练：当前 curated case 已建好与可导出，但还未并入训练脚本。
  - [ ] 性能对标：初步目标是超越人类新手水平，最终目标是达到或超越熟手或者专家水平。

## 📊 算法细节 (Algorithm Details)
- **State Representation**:
  - 旧版特征为 142 维。
  - PPO 路线使用 `state_to_vector_new()`，输入维度为 **562**。
  - 当前 MCTS + NN 主线路线使用 `state_to_vector_np()`，输入维度为 **567**。
  - 包含：工厂状态（Factories）、中心区统计（Center）、玩家墙面（Wall）、模式行（Pattern Lines）、分数与地板惩罚。
  - 特点：消除颜色与位置的伪线性关系，增强模型对非线性决策空间的感知。

- **Training Objective**:
  - - **Policy Loss**: 使用 **Cross Entropy (交叉熵)** 拟合 MCTS 搜索产生的访问频率分布 $\pi$。
    - 公式: $L_{pi} = -\sum \pi \log(p)$
  - **Value Loss**: 使用 均方误差（MSE）拟合对局最终结果 $z$（$z \in [-1, 1]$）。
  - 当前训练脚本中，value loss 默认降低权重使用：
    - `loss = policy_loss + 0.1 * value_loss`
  - 原因：value 头在早期训练中更容易不稳定，降低权重有助于先让 policy 收敛。

## 🧩 关键文件
- `logic.py`: 游戏规则、结算、合法动作、状态编码。
- `explore_mtcs.py`: 当前主用 MCTS 实现，包含 policy/value 评估与回传。
- `azul_transformer.py`: 当前主用 transformer policy-value 网络。
- `azul_net.py`: 较早期的 MLP 双头网络实现，仍保留作兼容与对照。
- `get_dataset.py`: 采集 MCTS self-play 数据或 Greedy 教师数据。
- `train_mcts_nn.py`: 训练 policy-value 网络。
- `run_iteration.py`: 执行一轮 self-play -> train -> arena -> promote。
- `loop_train.py`: 批量循环运行多轮迭代。
- `curated_cases.py`: 人工整理的关键局面回归集。
- `check_curated_cases.py`: 检查当前模型是否能解出 curated cases。
- `build_curated_dataset.py`: 从 curated cases 导出小型监督数据集。
- `battle.py`: 多局对战测试脚本。
- `server.py`: 使用网络和AI在Unity种进行对战。

## 🔧 常用命令
- 采集 Greedy 教师数据：
```bash
python get_dataset.py --mode greedy --games 1000 --output greedy_teacher_dataset.pkl
```

- 用指定模型采集 MCTS self-play 数据：
```bash
python get_dataset.py --mode mcts --games 300 --output mcts_v4_selfplay.pkl --sims 200 --model azul_net_v4.pt
```

- 训练新模型：
  - 在 `train_mcts_nn.py` 中修改 `data_path` 和 `save_path`，例如从 `greedy_teacher_dataset.pkl` 训练 `azul_net_v4.pt`，或从 `mcts_v4_selfplay.pkl` 训练 `azul_net_v5.pt`。

- 对战测试：
  - 在 `battle.py` 中指定模型路径、对手类型、`n_simulations` 后运行：
```bash
python battle.py
```
- 与Unity互动（上方链接提供的仓库）：
  - 默认使用 transformer champion，直接运行：
```bash
python server.py
```
## Transformer 主线（2026-05 / 2026-06）

当前 champion：

```bash
models/transformer_champion.pt
```

执行一轮 self-play -> train -> arena：

```bash
python run_iteration.py --allow-promote
```

连续跑多轮迭代：

```bash
python loop_train.py --iterations 10 --pause-seconds 5 --allow-promote
```

常用覆盖参数：

```bash
python run_iteration.py --games 300 --selfplay-sims 100 --train-epochs 6 --arena-games-per-side 10
```

当前补充说明：

- 当前 `server.py` 默认模型：`models/transformer_champion.pt`
- 当前已知 champion 来源：`transformer_candidate_20260519_213341.pt`
- curated cases 当前通过数：`2/3`
- curated dataset 已可导出到 `artifacts/curated_positions/`，但尚未正式接入训练流程

Replay 持久化：

- Each `run_iteration.py` call saves self-play training data locally as `replays/selfplay_<timestamp>.pkl`.
- The loop then trains from the most recent replay files selected by `--replay-window`.
- A lightweight manifest is appended to `replays/manifest.jsonl` so you can audit what each loop iteration generated.

代码状态：

- Active transformer path stays at repo root: `run_iteration.py`, `loop_train.py`, `get_dataset.py`, `train_mcts_nn.py`, `battle.py`, `server.py`.
- Deprecated PPO/BC experiments now live under `legacy/ppo_bc/`.
- Old root commands such as `train.py`, `train_bc.py`, `bc_to_ppo.py`, and `enjoy.py` are temporary compatibility wrappers and should not receive new work.
