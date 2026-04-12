import random
import torch
import numpy as np
from config import *  # 导入你之前定义的颜色常量
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None
from abandon_teach import BCPolicy
from train_scorer import ScoreModel  # 改成你的真实文件名


class HumanAgent:
    def decide(self, game):
        legal_moves = game.get_legal_moves()

        while True:  # 用循环包裹，直到输入正确为止
            print("\n--- 轮到你了，人类选手 ---")
            print(game.display_all_info())
            print("合法动作参考 (部分):", legal_moves[:5], "..." if len(legal_moves) > 5 else "")

            try:
                raw_input = input("请输入 [来源 颜色 目标行] (例如: 0 1 2 或 center 3 5): ").strip().split()

                if len(raw_input) != 3:
                    print("⚠️ 格式错误！请确保输入三个参数。")
                    continue

                # 处理来源：如果是数字就转int，否则保留字符串
                src = raw_input[0]
                if src.isdigit():
                    src = int(src)

                # 颜色和目标行永远是数字
                col = int(raw_input[1])
                target = int(raw_input[2])

                choice = (src, col, target)

                if choice in legal_moves:
                    return choice
                else:
                    print(f"❌ 非法动作 {choice}！请查看规则或面板状态。")

            except ValueError:
                print("⚠️ 输入包含非数字！(除了 'center' 以外都应该是数字)")


class RandomAgent:
    def decide(self, game):
        legal_moves = game.get_legal_moves()
        # 随机抓一个动作，这就是 AI 的“思考”
        return random.choice(legal_moves)


class GreedyAgent:
    def decide(self, game):
        legal_moves = game.get_legal_moves()
        best_move = None
        best_score_diff = -999  # 初始设为一个极小值

        for move in legal_moves:
            # 1. 克隆一个虚拟的游戏分身（防止弄乱真实的棋盘）
            # 注意：这里需要你实现一个简单的拷贝逻辑，或者手动模拟

            # 2. 模拟执行这个 move
            # 3. 计算这一步产生的即时收益 = (拿到的砖进入 pattern_lines 的分) - (掉进地板的扣分)

            # 简单版评估函数：
            score_gain = self.evaluate_move(game, move)

            if score_gain > best_score_diff:
                best_score_diff = score_gain
                best_move = move

        return best_move if best_move else random.choice(legal_moves)

    def evaluate_move(self, game, move):
        source, color, row_idx = move
        player = game.players[game.current_player_idx]

        # 1. 计算这步能拿多少砖
        count, _ = game.public_board.preview_pick(source, color)  # 增加一个只看拿多少但不真拿的函数

        # 2. 如果是扔进地板 (row_idx == 5)
        if row_idx == 5:
            return -2 * count  # 惩罚：每块扣2分

        # 3. 如果是放进 Pattern Lines
        line = player.pattern_lines[row_idx]
        spaces_left = (row_idx + 1) - line.count(color)  # 还能放几个

        actual_placed = min(count, spaces_left)
        overflow = max(0, count - spaces_left)

        score = 0
        # 价值 A：放进去的砖越多越好（特别是快填满时）
        score += actual_placed * 2

        # 价值 B：如果这一步直接填满了，给个大奖励
        if actual_placed == spaces_left:
            score += 10
            # 如果填满的这行在墙上能连号，再加分
            col_idx = color_to_column(row_idx, color)
            score += player.calculate_move_score(row_idx, col_idx) * 5

        # 价值 C：扣除地板分
        score -= overflow * 4

        return score


class PPOAgent:
    def __init__(self, path):
        if MaskablePPO is None:
            raise ImportError("sb3_contrib is required to use PPOAgent")
        self.model = MaskablePPO.load(path)

    def decide(self, game):
        # 🌟 关键：获取当前玩家的视角数据
        curr_id = game.current_player_idx
        obs_dict = game.get_observation_for_player(curr_id)  # 确保这里的逻辑和 Env 训练时一致
        obs_vector = game.state_to_vector(obs_dict)

        # 2. 生成符合规格的 NumPy Mask
        mask = self.get_refined_mask(game)

        # 3. 预测动作
        # deterministic=True 很关键，因为作为陪练，它应该表现得“稳定且专业”
        action_idx, _ = self.model.predict(
            obs_vector,
            action_masks=mask,
            deterministic=True
        )

        # 返回原始动作对象（如 (factory_idx, color, row_idx)）
        return ACTION_LOOKUP[action_idx]

    def get_refined_mask(self, game):
        # 初始化全为 False 的 180 维布尔数组
        mask = np.zeros(180, dtype=bool)
        legal_moves = game.get_legal_moves()

        for move in legal_moves:
            if move in REVERSE_LOOKUP:
                idx = REVERSE_LOOKUP[move]
                mask[idx] = True
        return mask


class BCAgent:
    def __init__(self, path, target_player_idx=0, device=None):
        self.target_player_idx = target_player_idx
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 仅仅是加载字典到内存
        checkpoint = torch.load(path, map_location=self.device)

        # 2. 动态获取维度（你之前的逻辑是对的）
        weight_keys = [k for k in checkpoint.keys() if "weight" in k]
        last_weight_key = sorted(weight_keys)[-1]
        obs_dim = checkpoint[weight_keys[0]].shape[1]
        action_dim = checkpoint[last_weight_key].shape[0]

        self.model = BCPolicy(obs_dim=obs_dim, action_dim=action_dim).to(self.device)

        # 3. 🌟 关键：执行真正的“权重灌入”
        # 既然之前报错没 key，那就尝试直接加载整个 checkpoint
        try:
            self.model.load_state_dict(checkpoint)
            # print("✅ 权重直接加载成功！")
        except Exception as e:
            # 如果还是不行，尝试提取一层
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                # print("✅ 从 'model_state_dict' 键加载成功！")
            else:
                # print(f"❌ 加载失败，请检查字典键名: {checkpoint.keys()}")
                raise e

        self.model.eval()

    def decide(self, game):
        state = game.get_observation_for_player(self.target_player_idx)
        obs = game.state_to_vector(state)
        mask = game.get_refined_mask()

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(obs_tensor)

            # 打印：当前合法动作的数量，看看是不是 0
            num_legal = torch.sum(mask_tensor).item()

            # 🌟 关键 Debug 信息
            masked_logits = logits.masked_fill(~mask_tensor, -1e9)  # 注意：这里取决于你的 mask 定义
            # 如果你的 mask 1是合法，0是非法，那么应该是填充 ~mask_tensor(即给0填-1e9)
            # 如果你之前去掉了 ~ 变成了 masked_fill(mask_tensor, -1e9)，那就把合法动作全屏蔽了！

            action_idx = masked_logits.argmax(dim=1).item()

            # # 打印 AI 的决策细节
            # print(f"[{'AI' if game.current_player_idx == 0 else 'OPP'}] "
            #       f"合法动作数: {num_legal}, 选定索引: {action_idx}, "
            #       f"Logit值: {logits[0, action_idx]:.2f}")
            #
            # if num_legal == 0:
            #     print("⚠️ 警告：竟然没有合法动作！这将导致死循环。")
        move = ACTION_LOOKUP[action_idx]
        count, _ = game.public_board.preview_pick(move[0], move[1])
        print(f"AI 决定: 从{move[0]}拿{move[1]}颜色{count}块，放进第{move[2]}行")
        return move


class ScoreAgent:
    def __init__(self, path, target_player_idx=0, device=None):
        self.target_player_idx = target_player_idx
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=self.device)

        self.model = ScoreModel(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def decide(self, game):
        # 1. 从固定玩家视角拿状态
        state = game.get_observation_for_player(self.target_player_idx)
        obs = game.state_to_vector(state)  # shape = (142,)

        # 2. 拿当前所有合法动作
        legal_moves = game.get_legal_moves()

        # 防御式写法：如果没有合法动作，直接报错
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # 3. 把当前 obs 复制成和合法动作数一样多
        #    比如有 12 个合法动作，就变成 (12, 142)
        obs_batch = np.repeat(obs[None, :], len(legal_moves), axis=0)

        # 4. 把合法动作映射成 action_idx
        action_idx_list = [REVERSE_LOOKUP[move] for move in legal_moves]
        action_idx_batch = np.array(action_idx_list, dtype=np.int64)

        # 5. 转成 PyTorch Tensor
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action_idx_batch, dtype=torch.long, device=self.device)

        # 6. 前向推理，得到每个合法动作的预测分数
        with torch.no_grad():
            pred_scores = self.model(obs_tensor, action_tensor)  # shape = (num_legal_moves,)

        # 7. 找分数最高的那个动作
        best_idx_in_legal = pred_scores.argmax().item()
        best_move = legal_moves[best_idx_in_legal]

        return best_move


