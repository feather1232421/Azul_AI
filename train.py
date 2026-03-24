import gymnasium as gym
from gymnasium import spaces
import numpy as np
from logic import AzulGame
from config import ACTION_LOOKUP
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import CheckpointCallback


from sb3_contrib import MaskablePPO # 换成这个
from sb3_contrib.common.maskable.utils import get_action_masks


class AzulEnv(gym.Env):
    def __init__(self):
        super(AzulEnv, self).__init__()
        # 动作空间：180个离散动作
        # 🌟 加上这一行，给计数器一个起点
        self.current_step = 0
        self.action_space = spaces.Discrete(180)
        # 观察空间：142个特征，范围根据你的向量实际情况定
        self.observation_space = spaces.Box(low=-50, high=200, shape=(142,), dtype=np.float32)

        self.game = AzulGame(num_players=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. 实例化游戏
        self.game = AzulGame(num_players=2)

        # 🌟 关键：调用你之前写的补砖函数！
        # 确保工厂里有砖，先手标志在中心
        self.game.start_round()

        # 如果你的逻辑里 start_round 包含了 refill 和确定先手，
        # 也可以直接调用 self.game.start_round()

        obs_dict = self.game.get_observation_current()
        obs_vector = self.game.state_to_vector(obs_dict)

        return obs_vector, {}

    def step(self, action_idx):
        move = ACTION_LOOKUP[action_idx]
        reward = 0
        # 🌟 既然有了 Mask，这里 move 必然合法
        acting_player_idx = self.game.current_player_idx
        acting_player = self.game.players[acting_player_idx]

        old_score = acting_player.score  # 记录动作前的分
        floor_before = len(acting_player.floor)

        # 1. 执行动作（只执行一次！）
        self.game.play_turn(*move)

        # 2. 计算即时反馈
        # floor_after = len(acting_player.floor)
        # 给一点点生存奖励 + 地板惩罚
        # reward = 0.1 - (floor_after - floor_before) * 0.5

        # 3. 回合结束逻辑（这部分你写得很棒，保持这样）
        # 3. 回合结束 → 计分
        if self.game.is_round_over():
            for player in self.game.players:
                old_score = player.score
                player.tiling_and_scoring(self.game.public_board.discard_pile)
                round_score = player.score - old_score

                if player.player_id == acting_player_idx:
                    reward += round_score

            if not self.game.is_game_over():
                self.game.start_round()
            else:
                for player in self.game.players:
                    old_score = player.score
                    player.endgame_scoring()
                    endgame_score = player.score - old_score

                    if player.player_id == acting_player_idx:
                        reward += endgame_score

        # 4. 更新时间
        self.current_step += 1
        terminated = self.game.is_game_over()
        truncated = self.current_step >= 200

        obs = self.game.state_to_vector(self.game.get_observation_current())
        return obs, reward, terminated, truncated, {}

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


if __name__ == "__main__":

    # 0. 🌟 关键的一步：先创建环境实例
    env = AzulEnv()

    # 1. 定义存档路径
    SAVE_PATH = "./models/"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 2. 检查是否有旧模型想接着练
    MODEL_FILE = "azul_ppo_v2.zip"

    if os.path.exists(MODEL_FILE):
        print(f"📖 发现旧模型 {MODEL_FILE}，正在读取并续练...")
        model = MaskablePPO.load(MODEL_FILE, env=env)
    else:
        print("✨ 未发现模型，正在初始化全新大脑...")
        model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./azul_logs/")

    # 3. 设置定时自动存档（每 10000 步存一次，防止停电或崩溃）
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=SAVE_PATH,
        name_prefix="azul_step"
    )

    # 4. 开始大练兵
    print("🚀 启动长线训练...")
    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback,
        reset_num_timesteps=False  # 🌟 关键：保持 TensorBoard 曲线连续
    )

    # 5. 最后保存一个终版
    model.save("azul_ppo_v3_final")
    print("✅ 训练圆满结束！")