import random
from config import *  # 导入你之前定义的颜色常量


class AzulGame:
    def __init__(self, num_players=2):
        # 实例化公共版图
        self.public_board = PublicBoard(num_players)

        # 实例化玩家版图（如果是2人局，列表里就有2个 PlayerBoard 对象）
        self.players = [PlayerBoard(i) for i in range(num_players)]

        # 记录轮到谁了
        self.current_player_idx = 0

    def start_round(self):
        # 喊公共版图去补货
        self.public_board.refill_factories()

    def get_current_player(self):
        return self.players[self.current_player_idx]


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

    def add_tiles_to_line(self, line_idx, color, count):
        # 将拿到的砖放入某一行，多余的掉到地板
        # 这里的逻辑：检查颜色是否匹配、是否已满、是否墙上已有该颜色...
        pass
