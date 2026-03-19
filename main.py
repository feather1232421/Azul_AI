from logic import AzulGame

if __name__ == "__main__":
    # 1. 创建游戏实例
    my_game = AzulGame()

    # 2. 执行发牌
    my_game.refill_factories()

    # 3. 打印场面
    my_game.display_status()