import time
import os
from sb3_contrib import MaskablePPO
from train import AzulEnv
from config import ACTION_LOOKUP


def enjoy():
    env = AzulEnv()
    # 🌟 自动寻找最新的模型
    model_path = "azul_ppo_v2.zip"  # 或者 models/ 里的最新文件
    if not os.path.exists(model_path):
        print("没找到模型文件！")
        return

    model = MaskablePPO.load(model_path, env=env)
    obs, _ = env.reset()

    print("🎮 AI 开始表演了...")
    for step_num in range(100):
        # 别忘了带上 Mask，不然 AI 可能会大脑宕机
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        move = ACTION_LOOKUP[action]
        obs, reward, terminated, truncated, _ = env.step(action)

        # --- 简单的棋盘渲染 ---
        print(f"\n第 {step_num + 1} 步: AI 选择从 {move[0]} 拿色，放入第 {move[1]} 行")
        print(f"当前得分: {env.game.players[0].score} | 奖励反馈: {reward:.2f}")

        # 如果你想看更详细的墙面，可以把 wall 打印出来
        # print(env.game.players[0].wall)

        if terminated or truncated:
            print("🏁 比赛结束！")
            break
        time.sleep(0.8)  # 留点时间让人眼观察


if __name__ == "__main__":
    enjoy()