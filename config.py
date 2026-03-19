# --- 颜色定义 ---
EMPTY = 0
BLUE = 1
YELLOW = 2
RED = 3
BLACK = 4
WHITE = 5
FIRST_PLAYER = 6

# --- 游戏参数 ---

# 不同人数对应的工厂数量映射表
PLAYER_FACTORY_MAP = {
    2: 5,
    3: 7,
    4: 9
}

TILES_PER_FACTORY = 4  # 每个板子上最多放4块砖
COLOR_COUNT = 5    # 除去先手牌共有5种颜色
BAG_INITIAL_COUNT = 20  # 每种颜色20块
