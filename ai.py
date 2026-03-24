import random
from config import *  # 导入你之前定义的颜色常量


class HumanAgent:
    def decide(self, game):
        legal_moves = game.get_legal_moves()

        while True:  # 用循环包裹，直到输入正确为止
            print("\n--- 轮到你了，人类选手 ---")
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
