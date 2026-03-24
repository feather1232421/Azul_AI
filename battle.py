from sb3_contrib import MaskablePPO
from train import AzulEnv
from config import *
from ai import GreedyAgent, RandomAgent


# 假设你的 Greedy 类定义在这里
# from agents import GreedyAgent

def battle():
    env = AzulEnv()
    model = MaskablePPO.load("models/azul_step_200000_steps.zip", env=env)
    greedy = GreedyAgent()  # 实例化你的 Greedy 类
    random = RandomAgent()

    ai_wins = 0
    total_games = 10

    print(f"🏁 开始 {total_games} 局对抗赛：PPO AI vs. Greedy")
    # print(f"🏁 开始 {total_games} 局对抗赛：PPO AI vs. Random")
    for g in range(total_games):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            # 获取当前是谁在下棋
            curr_player_idx = env.game.current_player_idx

            if curr_player_idx == 0:
                # --- AI 回合 ---
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                # --- Greedy/Random 回合 ---
                # 1. 让你的 Greedy/Random 类计算出 move 元组
                move_tuple = greedy.decide(env.game)
                # move_tuple = random.decide(env.game)
                # 2. 翻译成 action_idx
                action = REVERSE_LOOKUP[move_tuple]

            obs, reward, terminated, truncated, _ = env.step(action)

        # 结算本局比分
        p0_score = env.game.players[0].score
        p1_score = env.game.players[1].score
        winner = "🤖 AI" if p0_score > p1_score else "🐍 Greedy"
        # winner = "🤖 AI" if p0_score > p1_score else "🐍 Random"
        if p0_score > p1_score: ai_wins += 1

        print(f"第 {g + 1} 局: AI {p0_score} : {p1_score} Greedy | 胜者: {winner}")
        # print(f"第 {g + 1} 局: AI {p0_score}: {p1_score} Random | 胜者: {winner}")

    print(f"\n最终胜率统计: AI 胜率 {(ai_wins / total_games) * 100:.1f}%")


if __name__ == "__main__":
    battle()
