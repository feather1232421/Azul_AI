import copy
from config import *
import numpy as np


class AzulSearchAgent:
    def __init__(self, evaluate_move_fn, top_k=10, verbose=False):
        self.evaluate_move_fn = evaluate_move_fn
        self.top_k = top_k
        self.verbose = verbose

    def clone_game(self, game):
        if hasattr(game, "clone"):
            return game.clone()
        return copy.deepcopy(game)

    def state_value(self, game, root_player_id):
        opp_id = 1 - root_player_id

        me = game.players[root_player_id]
        opp = game.players[opp_id]

        score_diff = me.score - opp.score

        my_bonus = me.player_board_bonus()
        opp_bonus = opp.player_board_bonus()

        return score_diff + 2.5 * my_bonus - 2.5 * opp_bonus

    def greedy_best_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in moves:
            score = self.evaluate_move_fn(game, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def decide(self, game):
        root_player_id = game.current_player_idx
        candidate_moves = game.get_search_moves_v2()

        if not candidate_moves:
            return None

        # 第一层还是可以先用 greedy 打分再筛 top-k
        scored_moves = []
        for move in candidate_moves:
            score = self.evaluate_move_fn(game, move)
            scored_moves.append((move, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_moves[: self.top_k]

        best_move = candidates[0][0]
        best_value = float("-inf")

        for move, greedy_score in candidates:
            sim_game = self.clone_game(game)
            sim_game.play_turn(*move)

            value = self.rollout_value(
                sim_game,
                root_player_id=root_player_id,
                depth_remaining=3  # 这里控制深度，意思是模拟场上所有人一共走这么多步之后的局面
            )

            if self.verbose:
                print(
                    f"move={move}, greedy_score={greedy_score:.4f}, "
                    f"rollout_value={value:.4f}"
                )

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def rollout_value(self, game, root_player_id, depth_remaining):
        # 终止条件
        if depth_remaining == 0 or game.is_game_over():
            return self.state_value(game, root_player_id)

        current_player = game.current_player_idx

        # 如果轮到 root 玩家：认真搜索候选动作，取最大值
        if current_player == root_player_id:
            candidate_moves = game.get_search_moves_v2()
            if not candidate_moves:
                return self.state_value(game, root_player_id)

            best_value = float("-inf")

            for move in candidate_moves:
                sim_game = self.clone_game(game)
                sim_game.play_turn(*move)
                if sim_game.is_round_over():
                    sim_game._internal_scoring_flow()

                value = self.rollout_value(sim_game, root_player_id, depth_remaining - 1)
                best_value = max(best_value, value)

            return best_value

        # 如果轮到对手：先用 greedy 近似
        else:
            opp_move = self.greedy_best_move(game)
            if opp_move is None:
                return self.state_value(game, root_player_id)

            sim_game = self.clone_game(game)
            sim_game.play_turn(*opp_move)
            if sim_game.is_round_over():
                sim_game._internal_scoring_flow()

            return self.rollout_value(sim_game, root_player_id, depth_remaining - 1)

    def evaluate_all_moves(self, game):
        # """
        # 专门为数据集收集设计的函数
        # 返回格式: {action_idx: rollout_value}
        # """
        root_player_id = game.current_player_idx
        # candidate_moves = game.get_search_moves_v2()
        candidate_moves = game.get_legal_moves()
        # 结果字典
        move_scores = {}

        if not candidate_moves:
            return move_scores

        # 1. 依然先用 greedy 粗筛，否则 180 个动作全做 rollout 太慢了
        scored_candidates = []
        for move in candidate_moves:
            score = self.evaluate_move_fn(game, move)
            scored_candidates.append((move, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        # top_k 可以设大一点，比如 10-15，让数据更丰富
        top_candidates = scored_candidates[:self.top_k]

        # 2. 对这些顶尖候选者进行深度推演 (Search)
        for move, _ in top_candidates:
            sim_game = self.clone_game(game)
            sim_game.play_turn(*move)

            # 这里的深度可以设为 1 (对应你之前的 3 步搜索逻辑)
            value = self.rollout_value(
                sim_game,
                root_player_id=root_player_id,
                depth_remaining=3
            )

            action_idx = REVERSE_LOOKUP[move]
            move_scores[action_idx] = value

        return move_scores

    # 未来扩展可用，暂时先占位
    # def rollout_value(game, root_player_id, depth):
    #     if depth == 0 or game.is_game_over():
    #         return evaluate_state(game, root_player_id)
    #
    #     current_player = game.current_player_idx
    #
    #     if current_player == root_player_id:
    #         # root玩家：认真搜索候选动作，取最好
    #         ...
    #     else:
    #         # 其他玩家：先用greedy/heuristic近似
    #         ...

    def evaluate_all_moves_refined(self, game):
        candidate_moves = game.get_search_moves_v2()
        full_scores = np.full(180, -100.0, dtype=np.float32)  # 默认惩罚分

        # 1. 先用 1 步 Greedy 给所有合法动作打个底分
        scored_candidates = []
        for move in candidate_moves:
            base_score = self.evaluate_move_fn(game, move)
            action_idx = REVERSE_LOOKUP[move]
            full_scores[action_idx] = base_score  # 填入底分
            scored_candidates.append((move, base_score))

        # 2. 选出最强的几个，进行昂贵的 3 步搜索覆盖
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        for move, _ in scored_candidates[:5]:
            sim_game = self.clone_game(game)
            sim_game.play_turn(*move)
            # 3步推演后的真实价值
            search_val = self.rollout_value(sim_game, game.current_player_idx, depth_remaining=1)
            full_scores[REVERSE_LOOKUP[move]] = search_val

        return full_scores




