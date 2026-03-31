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

TILES_PER_FACTORY = 4   # 每个板子上最多放4块砖
COLOR_COUNT = 5         # 除去先手牌共有5种颜色
BAG_INITIAL_COUNT = 20  # 每种颜色20块


# 生成所有可能的组合 (6个来源 x 5个颜色 x 6个目标行)
ACTION_LOOKUP = []
for src in ["center", 0, 1, 2, 3, 4]:
    for col in range(1, 6):  # 颜色 1-5
        for row in range(0, 6):  # 行 0-4 是墙，5 是地板
            ACTION_LOOKUP.append((src, col, row))


# 建立反向索引：输入元组，返回 0-179 的整数
REVERSE_LOOKUP = {move: i for i, move in enumerate(ACTION_LOOKUP)}
# move 是一个(src, col, target)的 tuple

# 这样：
# 索引 0 代表 ("center", 1, 0)
# 索引 1 代表 ("center", 1, 1)
# ...以此类推
color_map = [
    [BLUE, YELLOW, RED, BLACK, WHITE],
    [WHITE, BLUE, YELLOW, RED, BLACK],
    [BLACK, WHITE, BLUE, YELLOW, RED],
    [RED, BLACK, WHITE, BLUE, YELLOW],
    [YELLOW, RED, BLACK, WHITE, BLUE],
]


def color_to_column(row_idx, color):
    """
    已知行号和颜色，计算在墙面 5x5 矩阵里的列号
    公式：(color_id - 1 + row_idx) % 5
    """
    return (color - 1 + row_idx) % 5
