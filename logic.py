import random
from config import *  # 导入你之前定义的颜色常量
import torch
import numpy as np
import copy


class AzulGame:
    def __init__(self, num_players=2, first_player=None, auto_init=True):
        self.num_players = num_players
        self.first_player = first_player
        self.next_round_first_player = None
        self.current_player_idx = None
        if auto_init:
            self._init_game()

    def _init_game(self):
        # 实例化公共版图
        self.public_board = PublicBoard(self.num_players)
        # 实例化玩家版图（如果是2人局，列表里就有2个 PlayerBoard 对象）
        self.players = [PlayerBoard(i) for i in range(self.num_players)]
        self.first_player = random.randint(0, self.num_players - 1)
        # 初始回合就该从首位开始
        self.current_player_idx = self.first_player
        self.next_round_first_player = None
        self.start_round()

    def reset(self):
        self._init_game()

    def advance_until_next_decision(self, agents_dict):
        # 自动运行游戏，直到轮到需要人工/AI决策的玩家，或者游戏结束。
        # agents_dict: 字典，key 是 player_id, value 是 agent 实例 (必须有 decide 方法)
        while not self.is_game_over():
            # 1. 检查当前玩家是否在“自动执行”名单里
            curr_id = self.current_player_idx
            # 如果当前玩家不在自动执行名单（比如是我们的 AI），停止并交还控制权
            if curr_id not in agents_dict:
                # 注意：如果这轮拿光了，得先结算才能判断下一轮谁先手
                if self.is_round_over():
                    self._internal_scoring_flow()
                    continue  # 结算完重新判定 current_player_idx
                break

            # 2. 处理轮次结束
            if self.is_round_over():
                self._internal_scoring_flow()
                continue

            # 3. 执行自动玩家的动作
            agent = agents_dict[curr_id]
            move = agent.decide(self)  # Greedy 就在这里发威
            # self.analyze_moves()
            self.play_turn(*move)

    def _internal_scoring_flow(self):
        """把计分和开新局的逻辑包起来"""
        for p in self.players:
            p.tiling_and_scoring(self.public_board.discard_pile)

        if not self.is_game_over():
            self.start_round()  # 重新补砖
        else:
            for p in self.players:
                p.endgame_scoring()

    def start_round(self):
        # 喊公共版图去补货
        self.public_board.refill_factories()
        if self.next_round_first_player:
            self.first_player = self.next_round_first_player
        self.next_round_first_player = None

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def play_turn(self, source, color, target_row):
        # player = self.players[self.current_player_idx]

        # 已经实现get_legal_move , 如果行动不是legal_move中的则会直接挡下，AI则是只会输出legal_move， 故此段检测废弃
        # # 1. 先问公共板：有货吗？
        # count, _ = self.public_board.preview_pick(source, color)
        # if count == 0:
        #     return False
        # assert count != 0, "PublicBoard check failed"
        # # 2. 再问玩家：能放吗？
        # if not player.can_accept(color, target_row):
        #     print("玩家表示：这砖我没法放这儿！")
        #     return False
        # assert player.can_accept(color, target_row), "PlayerBoard check failed"
        #
        # # 3. 检查都过了，正式开始“原子化”操作

        # ... 执行真正的 pick 和 add ...
        self._apply_move(source, color, target_row)
        if self.is_round_over():
            self._internal_scoring_flow()

    def _apply_move(self,source, color, target_row):
        player = self.players[self.current_player_idx]
        # 1. 取砖逻辑不变
        count, got_first = self.public_board.pick_tiles(source, color)

        if got_first:
            player.receive_first_player_marker()
            self.next_round_first_player = self.current_player_idx

        # 2. 【关键修复】判定 target_row
        if target_row == 5:
            # 如果是 5，代表 AI 决定直接扔进地板
            player.floor.extend([color] * count)
        else:
            # 否则，才是正常的放砖逻辑
            player.add_tiles_to_line(target_row, color, count)
        # 换下一个人
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        return True

    def is_round_over(self):
        return self.public_board.is_empty()

    def is_game_over(self):
        for player in self.players:
            for row in range(5):
                # 如果某一行全为 True，说明连成一线了
                if all(player.wall[row]):
                    return True
        return False

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

    def display_all_info(self):
        print("\n" + "=" * 30)
        print(f"🚩 当前回合：玩家 {self.current_player_idx}")
        print("=" * 30)

        self.public_board.display_status()
        for i in range(self.num_players):
            # 如果是当前玩家，加个小星星标识
            prefix = "⭐️ " if i == self.current_player_idx else "  "
            print(f"{prefix}玩家 {i} 的状态：")
            self.players[i].display_board()

    def get_observation_current(self):
        # 确定当前轮到谁
        me_idx = self.current_player_idx

        # 重新排列玩家列表：第一个永远是“我”，后面是“其他人”
        # 比如 3 人局，轮到 1 号动，列表就是 [Player1, Player2, Player0]
        sorted_players = [self.players[me_idx]] + \
                         [self.players[i] for i in range(self.num_players) if i != me_idx]

        state = {
            "factories": self.public_board.factories,
            "center": self.public_board.center,
            "me": sorted_players[0].to_dict(),  # 给 PlayerBoard 加个转字典的方法
            "opponents": [p.to_dict() for p in sorted_players[1:]]
        }
        return state

    # 在 logic.py 中增加这个方法，支持指定视角
    def get_observation_for_player(self, target_player_idx):
        # 第一个永远是指定的 target_player
        sorted_players = [self.players[target_player_idx]] + \
                         [self.players[i] for i in range(self.num_players) if i != target_player_idx]

        state = {
            "factories": self.public_board.factories,
            "center": self.public_board.center,
            "me": sorted_players[0].to_dict(),
            "opponents": [p.to_dict() for p in sorted_players[1:]]
        }
        return state

    def state_to_vector_np(sefl, state):
        # 预分配 567 维空间
        vec = np.zeros(567, dtype=np.float32)
        ptr = 0

        # --- 1. 工厂 (120维) ---
        # 每个工厂 4 格，每格 6 位 one-hot
        factories = np.array(state['factories'])  # 假设是 (5, 4)
        for tile in factories.flatten():
            # tile 是 0-5，直接在偏移量上填 1
            vec[ptr + tile] = 1.0
            ptr += 6

        # --- 2. 中心 (6维) ---
        center = state['center']
        for tile in center:
            if 1 <= tile <= 6:
                vec[ptr + tile - 1] += 1.0  # 统计法，不需要 one-hot
        ptr += 6

        # --- 3. 我方状态 ---
        me = state['me']

        # 墙面 (25维): 直接用 flatten 覆盖
        wall = np.array(me['wall'], dtype=np.float32).flatten()
        vec[ptr: ptr + 25] = wall
        ptr += 25

        # 准备区 (150维): 5行 * 5格 * 6(one-hot)
        for line in me['pattern_lines']:
            # 补齐到 5 格
            padded = line + [0] * (5 - len(line))
            for tile in padded:
                vec[ptr + tile] = 1.0
                ptr += 6

        # 分数归一化 (1维)
        vec[ptr] = me['score'] / 150.0
        ptr += 1

        # 地板 (42维): 7格 * 6(one-hot)
        floor = (me['floor'] + [0] * 7)[:7]
        for tile in floor:
            vec[ptr + tile] = 1.0
            ptr += 6

        # --- 4. 对方状态 (同样逻辑，省略重复部分，假设偏移量对齐) ---
        # ... 这里重复一遍 me 的逻辑填充对方数据 ...
        # --- 4. 对方状态 ---
        for opp in state['opponents']:
            opp_wall = np.array(opp['wall'], dtype=np.float32).flatten()
            vec[ptr: ptr + 25] = opp_wall
            ptr += 25

            for line in opp['pattern_lines']:
                padded = line + [0] * (5 - len(line))
                for tile in padded:
                    vec[ptr + tile] = 1.0
                    ptr += 6

            vec[ptr] = opp['score'] / 150.0
            ptr += 1

            floor = (opp['floor'] + [0] * 7)[:7]
            for tile in floor:
                vec[ptr + tile] = 1.0
                ptr += 6
        # 然后再填 5 维灵魂特征
        ptr = 562

        # --- 5. 灵魂特征 (新增的 5 维) ---
        # 先手标志 (假设 6 代表 1号标)
        vec[ptr] = 1.0 if 6 in me['floor'] else 0.0
        vec[ptr + 1] = 1.0 if any(6 in opp['floor'] for opp in state['opponents']) else 0.0

        # 进度感知
        vec[ptr + 2] = sum(row.count(True) for row in me['wall']) / 25.0
        # 对方进度也得看，不然不知道对面要赢了
        opp_wall = state['opponents'][0]['wall']
        vec[ptr + 3] = sum(row.count(True) for row in opp_wall) / 25.0

        # 分差
        vec[ptr + 4] = (me['score'] - state['opponents'][0]['score']) / 50.0

        return vec

    def state_to_vector_new(self, state):
        features = []
        # 1. 工厂：5个工厂 * 4个格子 * 6(one-hot) = 120个数字（原来是20个）
        for factory in state['factories']:
            for tile in factory:
                features.extend(color_to_onehot(tile))  # 0=空也有自己的one-hot位
        # 2. 中心：统计法，本来就对，不用改
        center_counts = [0] * 6
        for tile in state['center']:
            center_counts[tile - 1] += 1
        features.extend(center_counts)
        # 3. "我"的状态
        me = state['me']
        # 墙面：已经是0/1，不用改
        for row in me['wall']:
            features.extend([1 if cell else 0 for cell in row])
        # 待修行行：每格颜色需要one-hot
        # 第i行最多i+1个格子，每格one-hot(6) -> 固定5行*5格*6 = 150个
        for i, line in enumerate(me['pattern_lines']):
            padded_line = line + [0] * (5 - len(line))  # 补0(空)到5格
            for tile in padded_line:
                features.extend(color_to_onehot(tile))
        # 分数
        features.append(me['score'])
        # 地板：补齐7位，每格one-hot(6) -> 42个（原来是7个）
        padded_floor = (me['floor'] + [0] * 7)[:7]
        for tile in padded_floor:
            features.extend(color_to_onehot(tile))
        # 4. 对手（同样逻辑）
        for opp in state['opponents']:
            for row in opp['wall']:
                features.extend([1 if cell else 0 for cell in row])
            for i, line in enumerate(opp['pattern_lines']):
                padded_line = line + [0] * (5 - len(line))
                for tile in padded_line:
                    features.extend(color_to_onehot(tile))
            features.append(opp['score'])
            padded_floor = (opp['floor'] + [0] * 7)[:7]
            for tile in padded_floor:
                features.extend(color_to_onehot(tile))
        return np.array(features, dtype=np.float32)

    def state_to_vector(self, state):
        # state 就是你刚才打印的那个大字典
        features = []

        # 1. 处理工厂 (Factories): 5个工厂 * 4个位置 = 20个数字
        for factory in state['factories']:
            features.extend(factory)

        # 2. 处理中心 (Center): 统计法
        # 因为中心区长度变动，AI 没法处理，所以我们统计 1-6 每种颜色有多少个
        center_counts = [0] * 6  # 索引0-4是颜色1-5，索引5是先手牌6
        for tile in state['center']:
            center_counts[tile - 1] += 1
        features.extend(center_counts)

        # 3. 处理“我” (Me):
        me = state['me']
        # 墙面 (Wall): 5x5 = 25个 0/1
        for row in me['wall']:
            features.extend([1 if cell else 0 for cell in row])
        # 待修行行 (Pattern Lines): 补齐成 5x5 = 25个数字
        for i, line in enumerate(me['pattern_lines']):
            # 这一行有几个，是什么颜色？
            # 补齐：比如第一行只有1位，补4个0
            padded_line = line + [0] * (5 - len(line))
            features.extend(padded_line)
        # 分数和地板 (Floor)
        features.append(me['score'])
        # 地板我们也补齐到 7 位（因为地板最多7位）
        padded_floor = (me['floor'] + [0] * 7)[:7]
        features.extend(padded_floor)

        # 4. 处理“对手” (Opponents):
        # 为了固定长度，我们假设最多支持 1 个对手（2人局）
        # 如果是多人局，就用循环处理前 N 个对手
        for opp in state['opponents']:
            # 同样的逻辑：墙面 + 行 + 分数 + 地板
            for row in opp['wall']:
                features.extend([1 if cell else 0 for cell in row])
            for i, line in enumerate(opp['pattern_lines']):
                padded_line = line + [0] * (5 - len(line))
                features.extend(padded_line)
            features.append(opp['score'])
            padded_floor = (opp['floor'] + [0] * 7)[:7]
            features.extend(padded_floor)

        # 最后转化成 NumPy 数组，这是转 PyTorch Tensor 的前置步
        return np.array(features, dtype=np.float32)

    def get_refined_mask(self):
        # 初始化全为 False 的 180 维布尔数组
        mask = np.zeros(180, dtype=bool)
        legal_moves = self.get_legal_moves()

        for move in legal_moves:
            if move in REVERSE_LOOKUP:
                idx = REVERSE_LOOKUP[move]
                mask[idx] = True
        return mask

    def clone(self):
        return copy.deepcopy(self)

    def analyze_moves(self):
        legal_moves = self.get_legal_moves()
        search_moves = self.get_search_moves_v2()
        pair_to_targets = {}
        for src, col, target in legal_moves:
            key = (src, col)
            if key not in pair_to_targets:
                pair_to_targets[key] = []
            pair_to_targets[key].append(target)

        print("total legal moves:", len(legal_moves))
        print("search moves:",len(search_moves))
        # print("num (src,col) pairs:", len(pair_to_targets))

        total_targets = 0
        for key, targets in pair_to_targets.items():
            total_targets += len(targets)
        #     print(key, "->", targets)

        # print("avg targets per (src,col):", total_targets / len(pair_to_targets))

    def get_search_moves_v2(self, keep_top_r=2):
        legal_moves = self.get_legal_moves()

        groups = {}
        for move in legal_moves:
            src, col, target = move
            key = (src, col)
            groups.setdefault(key, []).append(move)

        search_moves = []

        for key, moves in groups.items():
            scored = []
            for move in moves:
                score = self.quick_target_score(move)
                scored.append((score, move))

            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [move for score, move in scored[:keep_top_r]]
            search_moves.extend(selected)

        return search_moves

    def quick_target_score(self, move):
        src, col, target = move
        num_tiles = self.count_tiles_taken(move)

        # floor 单独处理
        if target == 5:
            return self.estimate_floor_value(move)

        # 下面这些逻辑你按自己的 board 结构改
        player = self.players[self.current_player_idx]  # 举例
        line = player.pattern_lines[target]  # 举例

        current_fill = len(line)  # 或你自己的写法
        capacity = target + 1

        # 这步放进去后，最多能塞多少
        placed = min(num_tiles, capacity - current_fill)
        overflow = max(0, num_tiles - (capacity - current_fill))

        # 越接近补满越好，overflow 越多越差
        fill_ratio = (current_fill + placed) / capacity

        score = 0
        score += 5 * fill_ratio
        score += 3 * placed
        score -= 4 * overflow

        # 如果刚好补满，再额外奖励
        if current_fill + placed == capacity:
            score += 4

        return score

    def count_tiles_taken(self, move):
        src, col, target = move

        if src == "center":
            tiles = self.public_board.center
        else:
            tiles = self.public_board.factories[src]

        return tiles.count(col)

    def estimate_floor_value(self, move):
        num_tiles = self.count_tiles_taken(move)

        if num_tiles <= 0:
            return -999  # 理论上不该出现，防一手

        penalty_table = [1, 3, 5, 7, 9, 12, 15]
        if num_tiles >= 7:
            floor_penalty = 15
        else:
            floor_penalty = penalty_table[num_tiles - 1]

        deny_bonus = 2 * num_tiles

        return deny_bonus - floor_penalty

    @classmethod
    def from_table_data(cls, table_data):
        game = cls(num_players=1 + len(table_data.opponents), auto_init=False)
        game.load_from_table_data(table_data)
        return game

    def load_from_table_data(self, table_data):
        self.current_player_idx = 0
        self.num_players = 1 + len(table_data.opponents)
        # 这里先只重建基础对象壳子
        self.public_board = PublicBoard(self.num_players)
        self.players = [PlayerBoard(i) for i in range(self.num_players)]

        self.public_board._load_factories(table_data.factories)
        self.public_board._load_center(table_data.center)
        self.players[0]._load_player(table_data.me)
        for i in range(len(table_data.opponents)):
            self.players[i+1]._load_player(table_data.opponents[i])

    def clone_for_search(self):
        new_game = AzulGame.__new__(AzulGame)
        new_game.num_players = self.num_players
        new_game.next_round_first_player = self.next_round_first_player
        new_game.current_player_idx = self.current_player_idx
        new_game.first_player = self.first_player
        new_game.public_board = self.public_board.clone_for_search()
        new_game.players = [p.clone_for_search() for p in self.players]
        return new_game

    # 二人局专用
    def get_game_result(self):
        score0 = self.players[0].look_score()
        score1 = self.players[1].look_score()

        if score0 > score1:
            return 0
        elif score1 > score0:
            return 1
        else:
            return -1


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
        self.center = [FIRST_PLAYER]
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
        # print("--- 布袋已空，正在回收弃牌堆 ---")
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

    # 在 PublicBoard 类内部添加
    def preview_pick(self, source, color):
        """
        预检：返回如果从 source 拿 color，能拿到多少块，以及是否含先手牌。
        此函数不修改任何成员变量（不 pop，不改 center）。
        """
        count = 0
        got_first = False

        if source == "center":
            # 1. 数颜色砖
            count = self.center.count(color)
            # 2. 检查是否有先手牌
            if FIRST_PLAYER in self.center:
                got_first = True
        else:
            # source 是工厂索引，直接看那个工厂里有多少个该颜色
            # 假设你的工厂是一个列表，如 [1, 1, 0, 2]
            factory = self.factories[source]
            count = factory.count(color)
            got_first = False  # 工厂里永远不会有先手牌

        return count, got_first

    def _load_factories(self, factories):
        for i in range(self.factory_count):
            for j in range(4):
                if factories[i][j].empty:
                    self.factories[i][j] = EMPTY
                else:
                    self.factories[i][j] = factories[i][j].color

    def _load_center(self, center):
        self.center = []
        for i in range(24):
            if center[i].empty:
                continue
            else:
                if center[i].color == 0:
                    self.center.append(FIRST_PLAYER)
                else:
                    self.center.append(center[i].color)

    def clone_for_search(self):
        new_publicboard = PublicBoard.__new__(PublicBoard)
        new_publicboard.factories = [f[:] for f in self.factories]
        new_publicboard.factory_count = self.factory_count
        new_publicboard.center = self.center[:]
        new_publicboard.bag = self.bag[:]
        new_publicboard.discard_pile = self.discard_pile[:]
        return new_publicboard


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

    def display_board(self):
        print(f"\n--- 玩家 {self.player_id} 的个人板图 ---")
        print("待修行 (Pattern Lines):")
        for i, line in enumerate(self.pattern_lines):
            # 用空格补齐，让它看起来像个三角形
            spaces = " " * (4 - i)
            print(f"{i}: {spaces}{line}")

        print("墙面 (Wall):")
        for row in self.wall:
            # 把 True 变成 'X'，False 变成 '.'
            row_str = " ".join(['X' if cell else '.' for cell in row])
            print(f"   [{row_str}]")

        print(f"地板扣分: {self.floor} | 当前得分: {self.score}")

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

    def calculate_move_score(self, row, col):
        """
        当一块砖贴在 (row, col) 时，计算它的即时得分
        """
        # 1. 横向扫描 (Horizontal)
        h_score = 1
        # 往左看
        c = col - 1
        while c >= 0 and self.wall[row][c]:
            h_score += 1
            c -= 1
        # 往右看
        c = col + 1
        while c < 5 and self.wall[row][c]:
            h_score += 1
            c += 1

        # 2. 纵向扫描 (Vertical)
        v_score = 1
        # 往上看
        r = row - 1
        while r >= 0 and self.wall[r][col]:
            v_score += 1
            r -= 1
        # 往下看
        r = row + 1
        while r < 5 and self.wall[r][col]:
            v_score += 1
            r += 1

        # 3. 特殊规则：
        # 如果横向和纵向都超过1块（即形成了十字），这块砖被算了两次，得分 = h + v
        # 如果只有孤零零的一块，得分 = 1
        if h_score > 1 and v_score > 1:
            return h_score + v_score
        else:
            # 如果只有单向连号，取大的那个（如果都不连，h和v都是1）
            return max(h_score, v_score)

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

                # 3. 【核心】计算得分
                # 再算连号分 (调用刚才写的那个函数)
                move_points = self.calculate_move_score(i, col_idx)
                self.score += move_points

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

    def endgame_scoring(self):
        # 填满横行 +2 分
        for i in range(5):
            for j in range(5):
                if not self.wall[i][j]:
                    break
            else:
                self.score += 2

        # 填满竖列 +7 分
        for i in range(5):
            for j in range(5):
                if not self.wall[j][i]:
                    break
            else:
                self.score += 7

        # --- 填满全部颜色 (+10 分) ---
        color_counts = [0] * 5  # 假设颜色编号是 1-5，对应索引 0-4
        for r in range(5):
            for c in range(5):
                if self.wall[r][c]:
                    # 获取该位置对应的颜色
                    # 提示：花砖的墙面颜色通常是 (c - r) % 5
                    # 或者你有一个现成的 color_map[r][c]
                    color = color_map[r][c]
                    color_counts[color - 1] += 1

        for count in color_counts:
            if count == 5:
                self.score += 10

    # 在 PlayerBoard 类中
    def can_accept(self, color, target_row):
        """
        玩家版图的自我检查：我不动数据，只告诉你行不行。
        """
        if target_row == 5:  # 地板永远能接
            return True

        line = self.pattern_lines[target_row]
        # 规则 1：颜色匹配或为空
        if any(t != EMPTY and t != color for t in line):
            return False
        # 规则 2：墙上没贴过
        col_idx = color_to_column(target_row, color)
        if self.wall[target_row][col_idx]:
            return False

        # 规则 3：这行没满（虽然满了也可以拿，但通常视为非法或直接掉地板，看你规则定死没）
        if EMPTY not in line:
            return False

        return True

    def to_dict(self):
        state = {
            "pattern_lines":self.pattern_lines,
            "wall": self.wall,
            "floor": self.floor,
            "score": self.score,
        }
        return state

    def player_board_bonus(self):
        bonus = 0.0

        # ===== pattern lines =====
        for row in self.pattern_lines:
            capacity = len(row)
            filled = 0

            for tile in row:
                if tile != EMPTY:
                    filled += 1

            if capacity > 0:
                ratio = filled / capacity
                bonus += ratio

            # 🔥 快完成奖励（很重要）
            if filled == capacity - 1 and capacity > 1:
                bonus += 0.8

            # 🔥 已完成奖励（但还没结算到墙）
            if filled == capacity:
                bonus += 1.5

        # ===== floor 惩罚 =====
        floor_penalty = len(self.floor)

        # 👉 Azul 实际是递增惩罚，这里简单模拟一下
        # 比如：1,2,3,4,5 → 惩罚越来越重
        bonus -= 0.5 * floor_penalty + 0.2 * (floor_penalty ** 2)

        return bonus

    def _load_player(self, player):
        # score
        self.score = player.score
        # wall
        for i in range(5):
            for j in range(5):
                if not player.coloredAreas[i][j].empty:
                    self.wall[i][j] = True
        # floor
        for i in range(7):
            if not player.loseAreas[i].empty:
                self.floor.append(player.loseAreas[i].color)

        # patten_line
        for i in range(5):
            for j in range(i+1):
                if not player.manualAreas[i][j].empty:
                    self.pattern_lines[i][j] = player.manualAreas[i][j].color

    def clone_for_search(self):
        new_playerboard = PlayerBoard.__new__(PlayerBoard)
        new_playerboard.player_id = self.player_id
        new_playerboard.score = self.score
        new_playerboard.wall = [row[:] for row in self.wall]
        new_playerboard.floor = self.floor[:]
        new_playerboard.pattern_lines = [line[:] for line in self.pattern_lines]
        return new_playerboard

    def look_score(self):
        return self.score

if __name__ == "__main__":
    game = AzulGame()
    game.analyze_moves()
