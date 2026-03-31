from config import *
import gymnasium as gym
from gymnasium import spaces
from logic import AzulGame
import random
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
alpha = 0.3
beta = 0.1


class AzulEnv(gym.Env):
    def __init__(self, opponents=None):
        super().__init__()
        # 动作空间：180个离散动作
        # 🌟 加上这一行，给计数器一个起点
        self.current_step = 0
        self.action_space = spaces.Discrete(180)
        # 观察空间：142个特征，范围根据你的向量实际情况定
        self.observation_space = spaces.Box(low=-50, high=200, shape=(142,), dtype=np.float32)
        self.game = AzulGame(num_players=2)
        # 建议：opponents 应该是模型对象的列表或字典
        self.opponents = opponents
        self.ai_player_id = None
        self.opp_player_id = None

    @property
    def is_training(self):
        return self.opponents is not None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.game.reset()
        # 🎲 关键：随机决定 AI 在这局游戏里是哪个 ID
        self.ai_player_id = random.choice([0, 1])
        self.opp_player_id = 1 - self.ai_player_id

        # 如果当前轮到的不是 AI，让对手先走
        if self.is_training:
            agents_dict = {self.opp_player_id: random.choice(self.opponents)}
            self.game.advance_until_next_decision(agents_dict)

        return self._get_obs(), {}

    def step(self, action_idx):
        # 获取 AI 及其对手
        ai_p = self.game.players[self.ai_player_id]
        opp_p = self.game.players[self.opp_player_id]

        reward = 0.0
        r_phi = 0.0
        r_score = 0.0
        r_terminal = 0.0
        r_first = 0.0
        self.current_step += 1

        # 1. 纯粹的动作执行
        move = ACTION_LOOKUP[action_idx]
        # 记录动作前的状态：中心盘是否有先手标记？
        marker_was_available = (FIRST_PLAYER in self.game.public_board.center)

        old_score_ai = ai_p.score
        old_score_opp = opp_p.score
        old_phi = self._potential(ai_p)
        self.game.play_turn(*move)
        phi_after_ai = self._potential(ai_p)

        r_phi = alpha * (phi_after_ai - old_phi)
        # reward += r_phi

        # 2. 策略分水岭
        if self.is_training:
            # 2. 自动快进到 AI 的下一个决策点
            agents_dict = {self.opp_player_id: random.choice(self.opponents)}
            self.game.advance_until_next_decision(agents_dict)
            # 3. 计算阶梯奖励
            score_after_ai = ai_p.score
            new_score_opp = opp_p.score
            # 分数差增量奖励 (降低权重，防止短视)
            reward = ((score_after_ai - old_score_ai) - (new_score_opp - old_score_opp)) / 10
            r_round_diff = beta * ((score_after_ai - new_score_opp) - (old_score_ai - old_score_opp))
            r_score = (score_after_ai - old_score_ai)
            # reward += r_score
            # reward += r_round_diff
            # 终局大奖 (赢球才是硬道理)
            if self.game.is_game_over():
                if ai_p.score > opp_p.score:
                    r_terminal = 1.0
                    reward += r_terminal
                elif ai_p.score < opp_p.score:
                    r_terminal = -1.0  # 输了也要罚，防止 AI 摆烂
                    reward += r_terminal

            if marker_was_available and not (FIRST_PLAYER in self.game.public_board.center):
                # 确认是 AI 拿走的（通常 play_turn 后标记会进当前玩家的地板）
                if self.game.next_round_first_player == self.ai_player_id:
                    r_first = 0.02
                    reward += r_first

        # 4. 准备返回
        obs = self._get_obs()
        terminated = self.game.is_game_over()
        truncated = self.current_step >= 200
        info = {
            "p0_score": self.game.players[0].score,
            "p1_score": self.game.players[1].score,
            "reward_total": reward,
            "r_phi": r_phi,
            "r_score": r_score,
            "r_first": r_first,
            "r_terminal": r_terminal,
            "phi_after_ai": phi_after_ai,
            "score_after_ai": score_after_ai,
            "final_ai_score": ai_p.score,
            "final_opp_score": opp_p.score,
            "r_round_diff": r_round_diff,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        # 暂时不需要画面，先 pass
        pass

    def close(self):
        # 暂时不需要清理，先 pass
        pass

    def action_masks(self):
        # 这就是你之前写的那个逻辑
        mask = [False] * 180
        legal_moves = self.game.get_legal_moves()
        for i, move in enumerate(ACTION_LOOKUP):
            if move in legal_moves:
                mask[i] = True
        return mask

    def _get_obs(self):
        # 强制以 AI 的 ID 来生成观察值，不管现在轮到谁，也不管游戏是否结束
        state = self.game.get_observation_for_player(self.ai_player_id)
        return self.game.state_to_vector(state)

    def _potential(self, player):
        progress = 0.0
        for line in player.pattern_lines:
            filled = 0
            for block in line:
                if block != EMPTY:
                    filled += 1
            length = len(line)
            progress += (filled / length) ** 2

        floor_risk = len(player.floor)

        return progress - 0.8 * floor_risk


class RewardDebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.r_phi = []
        self.r_score = []
        self.r_terminal = []
        self.r_round_diff = []

    def _on_step(self):
        infos = self.locals["infos"]

        for info in infos:
            if "r_phi" in info:
                self.r_phi.append(info["r_phi"])
                self.r_score.append(info["r_score"])
                self.r_terminal.append(info["r_terminal"])
                self.r_round_diff.append(info["r_round_diff"])

        return True

    def _on_rollout_end(self):
        if len(self.r_phi) > 0:
            print("\n[DEBUG REWARD]")
            print(f"phi_mean: {np.mean(self.r_phi):.3f}")
            print(f"score_mean: {np.mean(self.r_score):.3f}")
            print(f"terminal_mean: {np.mean(self.r_terminal):.3f}")
            print(f"r_round_diff: {np.mean(self.r_round_diff):.3f}")
            self.r_phi.clear()
            self.r_score.clear()
            self.r_terminal.clear()
            self.r_round_diff.clear()
