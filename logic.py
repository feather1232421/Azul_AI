import random
from config import *  # 导入你之前定义的颜色常量


class AzulGame:
    def __init__(self, num_players=2, first_player=None):
        # 实例化公共版图
        self.public_board = PublicBoard(num_players)

        # 实例化玩家版图（如果是2人局，列表里就有2个 PlayerBoard 对象）
        self.players = [PlayerBoard(i) for i in range(num_players)]
        if first_player is not None:
            self.first_player = first_player
        else:
            # 假设玩家编号是 0 和 1 (符合编程习惯)
            # randint(0, 1) 会随机返回 0 或 1
            self.first_player = random.randint(0, num_players - 1)

            # 既然随机出了首通玩家，初始回合就该从他开始
        self.current_player_idx = self.first_player
        self.next_round_first_player = None

    def start_round(self):
        # 喊公共版图去补货
        self.public_board.refill_factories()

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def play_turn(self, source, color, target_row):
        """
        MVP 核心动作：
        source: 工厂索引 (0-4) 或 "center"
        color: 颜色编号 (1-5)
        target_row: 玩家想放哪一行 (0-4)
        """
        player = self.players[self.current_player_idx]

        # 1. 从公共区拿砖
        count, got_first = self.public_board.pick_tiles(source, color)

        if count == 0:
            print("⚠ 那个位置没有这种颜色的砖！请重新输入。")
            return False

        # 2. 处理先手牌
        if got_first:
            player.receive_first_player_marker()
            self.next_round_first_player = self.current_player_idx
            print(f"✨ 玩家 {self.current_player_idx} 拿到了先手标记！")

        # 3. 放入玩家版图
        player.add_tiles_to_line(target_row, color, count)

        # 4. 换下一个人
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        return True

    def is_round_over(self):
        return self.public_board.is_empty()

    def get_legal_moves(self):
        moves = []
        curr_player = self.players[self.current_player_idx]

        # 1. 搜集所有有砖的来源 (工厂 + 中心)
        sources = []
        for i, f in enumerate(self.public_board.factories):
            if any(slot != EMPTY for slot in f):
                sources.append(i)
        if self.public_board.center:
            # 注意：中心区可能有先手牌(6)，我们要滤掉，只看1-5号色
            if any(t in range(1, 6) for t in self.public_board.center):
                sources.append("center")

        # 2. 遍历来源和颜色
        for src in sources:
            # 获取该来源中所有的颜色种类
            if src == "center":
                available_colors = set(t for t in self.public_board.center if t in range(1, 6))
            else:
                available_colors = set(t for t in self.public_board.factories[src] if t != EMPTY)

            for col in available_colors:
                # 3. 检查玩家的 5 行哪行能放
                for row_idx in range(5):
                    # 核心判定：
                    # A. 这一行是空的，或者颜色匹配
                    # B. 墙上这一行还没贴过这个颜色
                    line = curr_player.pattern_lines[row_idx]
                    is_empty_or_same = all(t == EMPTY or t == col for t in line)
                    is_not_on_wall = not curr_player.wall[row_idx][color_to_column(row_idx, col)]

                    # 特殊：如果这一行已经满了，也不能放
                    is_not_full = EMPTY in line

                    if is_empty_or_same and is_not_on_wall and is_not_full:
                        moves.append((src, col, row_idx))

                # 永远可以选：直接扔进地板 (在我们的逻辑里，这通常通过一个特殊的 row_idx 来表示，比如 5)
                moves.append((src, col, 5))

        return moves


class PublicBoard:
    def __init__(self, num_players=2):

        # 1. 初始化布袋 (Bag)
        self.bag = []
        self._init_bag()

        # 2. 初始化工厂盘 (Factories)
        # 5个工厂，每个工厂4个空格，初始全为空 (EMPTY=0)
        # 自动根据人数设定工厂数
        self.num_players = num_players
        self.factory_count = PLAYER_FACTORY_MAP.get(num_players, 5)  # 如果没查到，默认5个

        # 剩下的初始化也要跟着变
        self.factories = [[EMPTY] * TILES_PER_FACTORY for _ in range(self.factory_count)]

        # 3. 初始化桌面中心 (Center Area)
        # 初始只有先手标记 (FIRST_PLAYER=6)
        self.center = [FIRST_PLAYER]

        # 4. 弃牌堆 (Discard Pile)
        self.discard_pile = []

    def _init_bag(self):
        # 填充布袋并洗牌
        # 1-5号颜色各 20 块
        for color in range(1, 6):
            self.bag.extend([color] * BAG_INITIAL_COUNT)
        random.shuffle(self.bag)

    def is_empty(self):
        # 检查公共板上是否已经没砖了（工厂和中心区全空）
        # 1. 检查所有工厂是否都是 [0, 0, 0, 0]
        # factories_empty 会是一个 True 或 False
        factories_empty = all(f == [EMPTY] * TILES_PER_FACTORY for f in self.factories)

        # 2. 检查中心区是否为空（除了可能的 0，但我们应该保证没有 0）
        center_empty = len(self.center) == 0

        # 只有两个都为空，才返回 True (代表这一轮拿光了)
        return factories_empty and center_empty

    def refill_factories(self):
        # 回合开始：从布袋给工厂补货
        for i in range(self.factory_count):
            for j in range(TILES_PER_FACTORY):
                if not self.bag:
                    # 如果袋子空了，把弃牌堆洗回去
                    self._recycle_discard()

                if self.bag:
                    self.factories[i][j] = self.bag.pop()
                else:
                    # 如果弃牌堆也是空的，那就真的没砖了（极端情况）
                    self.factories[i][j] = EMPTY

    def _recycle_discard(self):
        # 回收弃牌堆逻辑
        print("--- 布袋已空，正在回收弃牌堆 ---")
        self.bag = self.discard_pile[:]
        self.discard_pile = []
        random.shuffle(self.bag)

    def display_status(self):
        # 在控制台打印当前场面，方便调试
        print("\n" + "=" * 20)
        print("当前工厂盘面：")
        for i, f in enumerate(self.factories):
            print(f"工厂 {i}: {f}")
        print(f"桌面中心: {self.center}")
        print(f"布袋剩余: {len(self.bag)} 块")
        print("=" * 20 + "\n")

    def pick_tiles(self, factory_idx, color):
        picked_count = 0
        got_first_player = False

        if factory_idx != "center":
            # --- 从工厂拿 ---
            current_factory = self.factories[factory_idx]
            for tile in current_factory:
                if tile == color:
                    picked_count += 1
                elif tile != EMPTY:  # 只有不是空的才扔进中心
                    self.center.append(tile)
            # 清空工厂
            self.factories[factory_idx] = [EMPTY] * TILES_PER_FACTORY

        else:
            # --- 从中心拿 ---
            # 1. 检查有没有先手牌
            if FIRST_PLAYER in self.center:
                self.center.remove(FIRST_PLAYER)
                got_first_player = True

            # 2. 计算拿到了多少目标颜色
            picked_count = self.center.count(color)

            # 3. 把剩下的（不是这个颜色的）留着，把要拿走的“过滤”掉
            self.center = [t for t in self.center if t != color]

        return picked_count, got_first_player


class PlayerBoard:
    def __init__(self, player_id):
        self.player_id = player_id
        self.score = 0

        # 待修行 (Pattern Lines): 5行，第一行1格，第五行5格
        # 我们用列表表示：[[0], [0,0], [0,0,0], ...]
        self.pattern_lines = [[EMPTY] * (i + 1) for i in range(5)]

        # 墙面 (Wall): 5x5 的矩阵，记录哪一格贴了砖
        self.wall = [[False] * 5 for _ in range(5)]

        # 地板/碎砖区 (Floor): 最多掉落 7 块
        self.floor = []

    def display_status(self):
        # 在控制台打印当前场面，方便调试
        print("\n" + "=" * 20)
        print(f"{self.player_id} 号玩家个人盘面：")
        for i, f in enumerate(self.pattern_lines):
            print(f"第 {i} 行: {f}")
        for i, f in enumerate(self.wall):
            print(f"第 {i} 行: {f}")
        print(f"个人地板: {self.floor}")
        print(f"个人分数: {self.score} 分")
        print("=" * 20 + "\n")

    def add_tiles_to_line(self, line_idx, color, count):
        # 1. 检查这一行是否已经有了别的颜色
        existing_colors = set(self.pattern_lines[line_idx]) - {EMPTY}
        is_wrong_color = len(existing_colors) > 0 and color not in existing_colors

        # 2. 检查墙上对应的位置是否已经有砖了 (这里假设你有一个映射表)
        # 墙上位置计算公式：(row + color_offset) % 5
        # 为了简单，我们先假设逻辑：is_on_wall = self.check_wall(line_idx, color)
        is_on_wall = self.wall[line_idx][color_to_column(line_idx, color)]

        # 3. 如果颜色不符或墙上已有，全部进地板
        if is_wrong_color or is_on_wall:
            self.floor.extend([color] * count)
            return

        # 4. 填充逻辑
        for i in range(len(self.pattern_lines[line_idx])):
            if self.pattern_lines[line_idx][i] == EMPTY and count > 0:
                self.pattern_lines[line_idx][i] = color
                count -= 1

        # 5. 剩下的全是碎砖
        if count > 0:
            self.floor.extend([color] * count)

    def receive_first_player_marker(self):
        # 玩家收到了先手牌，自己处理它（通常是丢进地板）
        self.floor.append(FIRST_PLAYER)
        # 以后如果你想加逻辑，比如“拿了先手牌会触发某个技能”，改这里就行

    def tiling_and_scoring(self, discard_pile):

        # discard_pile: 传入公共区的弃牌堆，要把剩下的砖扔回去

        for i in range(5):
            # 1. 检查这一行是否满了（最后一个位置不是 EMPTY）
            row = self.pattern_lines[i]
            if EMPTY not in row:
                color = row[0]
                col_idx = color_to_column(i, color)

                # 2. 贴砖到墙上
                self.wall[i][col_idx] = True

                # 3. 【核心】计算得分（暂时只算基础分，后面教你算连号分）
                self.score += 1

                # 4. 清理当前行：留下一块，剩下的进弃牌堆
                # 这一行除了被拿走的那块，剩下的都要回收
                discard_pile.extend([color] * (len(row) - 1))
                self.pattern_lines[i] = [EMPTY] * (i + 1)  # 重置为空行

        # 5. 处理地板扣分
        penalty = [1, 1, 2, 2, 2, 3, 3]  # 规则：前两块扣1分，以此类推
        for idx, tile in enumerate(self.floor):
            if idx < len(penalty):
                self.score -= penalty[idx]
            # 碎砖（除了先手牌）都要回弃牌堆
            if tile != FIRST_PLAYER:
                discard_pile.append(tile)

        # 清空地板和分数的下限检查
        self.floor = []
        self.score = max(0, self.score)




