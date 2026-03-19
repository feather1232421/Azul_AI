from logic import AzulGame


def main():
    # 1. 初始化游戏
    game = AzulGame(num_players=2)
    print("🎮 Azul AI 引擎 (MVP版) 启动！")

    # 2. 回合开始：发牌
    game.public_board.refill_factories()

    # 3. 游戏主循环 (直到这轮砖被拿光)
    while not game.public_board.is_empty():
        game.public_board.display_status()
        curr_idx = game.current_player_idx
        print(f"👉 当前回合：玩家 {curr_idx}")

        try:
            # 这里的输入约定：工厂号(-1代表中心) 颜色 行号
            user_input = input("请输入动作 [工厂号(-1为中心) 颜色 行号]: ")
            f_idx, col, row = map(int, user_input.split())

            # 转换 -1 为 "center"
            source = "center" if f_idx == -1 else f_idx

            # 执行动作
            success = game.play_turn(source, col, row)

            if success:
                # 打印一下该玩家现在的 Pattern Lines 状态
                print(f"✅ 玩家 {curr_idx} 的待修行行：{game.players[curr_idx].pattern_lines}")
                print(f"📉 地板扣分区：{game.players[curr_idx].floor}")

        except Exception as e:
            print(f"❌ 输入错误或非法操作: {e}，请重新输入！")

    print("\n🏁 这一轮的砖拿光了！MVP 流程演示结束。")


if __name__ == "__main__":
    main()