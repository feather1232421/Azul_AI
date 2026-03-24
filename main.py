from logic import AzulGame
from ai import RandomAgent, GreedyAgent, HumanAgent


def main():
    game = AzulGame(num_players=2)
    agent1 = HumanAgent()
    agent2 = GreedyAgent()
    round_count = 1

    while not game.is_game_over():
        print(f"\n第 {round_count} 轮开始！")
        game.start_round()
        print(game.get_observation_current())
        sample_vector = game.state_to_vector(game.get_observation_current())
        print(f"我的 AI 视网膜长度是: {len(sample_vector)}")
        print("合法的行动有：")
        print(game.get_legal_moves())
        # 拿砖阶段
        while not game.public_board.is_empty():
            curr_player_idx = game.current_player_idx
            agent = agent1 if curr_player_idx == 0 else agent2

            # 1. 记录动作前状态 (可选)
            if curr_player_idx == 0:
                game.display_all_info()
            action = agent.decide(game)
            f_idx, col, row = action

            # 2. 格式化打印动作
            source_name = f"工厂 {f_idx}" if isinstance(f_idx, int) else "桌面中心"
            row_name = f"第 {row} 行" if row < 5 else "掉落地板"
            print(f"🤖 玩家 {curr_player_idx} 执行动作: 从 [{source_name}] 拿颜色 {col} 放入 [{row_name}]")

            # 3. 执行
            game.play_turn(*action)

            # 4. 打印即时反馈
            p = game.players[curr_player_idx]
            print(f"   此时该行状态: {p.pattern_lines[row] if row < 5 else p.floor}")

        # 计分阶段
        print(f"--- 第 {round_count} 轮结束，开始计分 ---")
        for p in game.players:
            p.tiling_and_scoring(game.public_board.discard_pile)
            print(f"玩家 {p.player_id} 当前总分: {p.score}")

        round_count += 1
        if round_count > 20:  # 安全阀，防止随机 AI 永远玩不完
            break
    for p in game.players:
        p.endgame_scoring()

    print("🏆 游戏正式结束！最终得分：", [p.score for p in game.players])


if __name__ == "__main__":
    main()
