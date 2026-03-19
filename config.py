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


def color_to_column(row_idx, color):
    """
    已知行号和颜色，计算在墙面 5x5 矩阵里的列号
    公式：(color_id - 1 + row_idx) % 5
    """
    return (color - 1 + row_idx) % 5
