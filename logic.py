import random
from config import *  # 导入你之前定义的颜色常量


class AzulGame:
    def __init__(self):
        # 1. 初始化布袋 (Bag)
        self.bag = []
        self._init_bag()

        # 2. 初始化工厂盘 (Factories)
        # 5个工厂，每个工厂4个空格，初始全为空 (EMPTY=0)
        self.factories = [[EMPTY] * TILES_PER_FACTORY for _ in range(FACTORY_COUNT)]

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
        for i in range(FACTORY_COUNT):
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