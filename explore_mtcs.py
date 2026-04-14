import math
import copy
import random
import time
import json
from ai import GreedyAgent
import torch
import torch.nn as nn
from logic import AzulGame
import pickle
from pathlib import Path
from config import *
import numpy as np
from azul_net import AzulNet


def compute_softmax_over_legal(policy_logits, legal_moves, temperature=1.0):
    priors = {}
    temperature = max(float(temperature), 1e-6)

    # 1. 取出合法动作对应的 logits
    logits = []
    for m in legal_moves:
        idx = REVERSE_LOOKUP[m]
        logits.append(policy_logits[idx].item() / temperature)

    # 2. 数值稳定的 softmax（减 max）
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits) + 1e-8

    # 3. 归一化，并映射回 move
    for m, e in zip(legal_moves, exp_logits):
        priors[m] = e / sum_exp

    return priors


def randomize_hidden_bag_for_search(game):
    search_game = game.clone_for_search()
    # Remaining bag composition is inferable, but the order is hidden.
    search_game.public_board.rng.shuffle(search_game.public_board.bag)
    return search_game


class MCTSNode:
    def __init__(self, game=None, parent=None, action=None, prior=0.0):
        self.children = []  # 存储子节点对象
        self.children_actions = {}  # 新增：用字典快速检索 {action: child_node}

        self.visits = 0
        self.wins = 0.0
        self.game = game  # clone过来的AzulGame实例
        self.parent = parent
        self.action = action  # (从哪拿, 颜色, 放哪里)
        self.player_idx = game.current_player_idx  # add_child之前记录
        self.prior = prior  # 网络给这个动作的先验概率

        self.untried_actions = self.game.get_legal_moves()  # 你现有的合法动作列表
        random.shuffle(self.untried_actions)  # 打乱顺序，避免偏差

        self.priors = None  # 存整张分布

    # def ucb_score(self, C=1.4):
    #     if self.visits == 0:
    #         return float('inf')
    #     q_value = self.wins / self.visits if self.visits > 0 else 0
    #     u_value = C * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
    #     return q_value + u_value

    # new
    def ucb_score(self, C=1.4):
        q_value = 0.0 if self.visits == 0 else self.wins / self.visits
        if self.parent is None:
            u_value = 0.0
        else:
            u_value = C * self.prior * math.sqrt(self.parent.visits + 1e-8) / (1 + self.visits)
        return q_value + u_value

    def is_expanded(self):
        return len(self.children) > 0

    def best_child(self, C=1.4):
        def score(child):
            # child.wins / child.visits 是“子节点行动方视角”的价值。
            # 对当前节点来说，走到这个 child 后轮到对手，因此 exploitation 要取反。
            q_value = 0.0 if child.visits == 0 else -(child.wins / child.visits)
            return q_value + C * child.prior * math.sqrt(self.visits + 1e-8) / (1 + child.visits)

        return max(self.children, key=score)

    # # new
    # def best_child(self, root_player_idx, C=1.4):
    #     if self.game.current_player_idx == root_player_idx:
    #         return max(self.children, key=lambda c: c.ucb_score(C))
    #     else:
    #         return min(self.children, key=lambda c: c.ucb_score(C))

    def add_child(self, action, prior):
        # 这里的 clone 只有在真正需要新节点时才做
        new_game = self.game.clone_for_search()
        new_game.play_turn(*action)

        child = MCTSNode(new_game, parent=self, action=action, prior=prior)
        self.children.append(child)
        self.children_actions[action] = child  # 记录动作
        return child


class MCTSAgent:
    def __init__(
        self,
        n_simulations=200,
        n_determinizations=4,
        my_player_idx=0,
        net=None,
        device=None,
        action_dim=180,
        use_policy=True,
        use_value=True,
        puct_c=1.4,
        prior_temperature=1.0,
        debug_log_path=None,
        debug_top_k=8,
        debug_label=None,
    ):
        self.n_simulations = n_simulations
        self.n_determinizations = max(1, n_determinizations)
        # 当前二人零和版本中，my_player_idx 不再参与 value 语义
        self.my_player_idx = my_player_idx
        self.device = device if device is not None else torch.device("cpu")
        self.net = net if net is not None else AzulNet()
        self.net.to(self.device)
        self.net.eval()
        self.action_dim = action_dim
        self.use_policy = use_policy
        self.use_value = use_value
        self.puct_c = puct_c
        self.prior_temperature = prior_temperature
        self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        self.debug_top_k = debug_top_k
        self.debug_label = debug_label or "mcts"
        self.debug_step = 0

    def _get_policy_obs_tensor(self, game):
        state = game.get_observation_current()
        obs = game.state_to_vector_np(state)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    # 二人零和，先这样
    def _get_value_obs_tensor(self, game):
        state = game.get_observation_current()
        obs = game.state_to_vector_np(state)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _evaluate_policy(self, game):
        obs = self._get_policy_obs_tensor(game)
        with torch.no_grad():
            policy_logits, _ = self.net(obs)
        return policy_logits.squeeze(0)

    def _evaluate_value(self, game):
        obs = self._get_value_obs_tensor(game)
        with torch.no_grad():
            _, value_logit = self.net(obs)
            value = torch.tanh(value_logit)
        return value.item()

    def _evaluate(self, game):
        obs = self._get_value_obs_tensor(game)
        with torch.no_grad():
            policy_logits, value_logit = self.net(obs)
            # value_logit 是一个标量，可以用 item()
            value = torch.tanh(value_logit).item() if self.use_value else 0.0

            # policy_logits 需要去掉 batch 维度，转回 CPU 上的 numpy 或 list
            # 这样你在 compute_softmax_over_legal 里访问比较快
            policy_vector = policy_logits.squeeze(0).cpu().numpy()

        return policy_vector, value

    def _get_priors(self, policy_logits, legal_moves):
        if self.use_policy:
            return compute_softmax_over_legal(
                policy_logits,
                legal_moves,
                temperature=self.prior_temperature,
            )
        uniform_prior = 1.0 / len(legal_moves)
        return {action: uniform_prior for action in legal_moves}

    def _build_mask(self, game):
        legal = game.get_legal_moves()
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for move in legal:
            idx = REVERSE_LOOKUP[move]
            mask[idx] = 1.0
        return legal, mask

    def _expand_root(self, root):
        policy_logits, _ = self._evaluate(root.game)
        legal_moves = root.game.get_legal_moves()
        priors = self._get_priors(policy_logits, legal_moves)
        for action, p in priors.items():
            root.add_child(action, prior=p)

    def _run_single_search(self, root_game, n_simulations):
        root = MCTSNode(root_game)
        self._expand_root(root)

        for _ in range(n_simulations):
            node = root
            while node.children and not node.game.is_game_over():
                node = node.best_child(C=self.puct_c)

            if node.game.is_game_over():
                value = self._terminal_value(node)
            else:
                policy_logits, value = self._evaluate(node.game)
                legal_moves = node.game.get_legal_moves()
                priors = self._get_priors(policy_logits, legal_moves)
                for action, p in priors.items():
                    node.add_child(action, prior=p)

            self._backprop(node, value)

        return root

    def _aggregate_roots(self, roots):
        total_visits = {}
        total_wins = {}
        total_priors = {}

        for root in roots:
            for child in root.children:
                action = child.action
                total_visits[action] = total_visits.get(action, 0) + child.visits
                total_wins[action] = total_wins.get(action, 0.0) + child.wins
                total_priors[action] = total_priors.get(action, 0.0) + child.prior

        return total_visits, total_wins, total_priors

    def _build_pi_from_visit_dict(self, visit_dict):
        pi = np.zeros(self.action_dim, dtype=np.float32)
        for action, visits in visit_dict.items():
            idx = REVERSE_LOOKUP[action]
            pi[idx] = visits
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def _write_debug_log(self, game, chosen_move, total_visits, total_wins, total_priors, roots):
        if self.debug_log_path is None:
            return

        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        root_values = []
        for root in roots:
            if root.children:
                root_values.append(max((-child.wins / child.visits) if child.visits > 0 else 0.0 for child in root.children))

        move_rows = []
        n_roots = max(len(roots), 1)
        for action, visits in sorted(total_visits.items(), key=lambda item: -item[1])[:self.debug_top_k]:
            wins = total_wins.get(action, 0.0)
            q_value = -(wins / visits) if visits > 0 else 0.0
            move_rows.append({
                "move": list(action),
                "visits": int(visits),
                "prior": float(total_priors.get(action, 0.0) / n_roots),
                "q": float(q_value),
            })

        payload = {
            "tag": f"{self.debug_label}_step_{self.debug_step}",
            "step": self.debug_step,
            "current_player": game.current_player_idx,
            "scores": [player.score for player in game.players],
            "n_simulations": self.n_simulations,
            "n_determinizations": self.n_determinizations,
            "legal_moves": int(len(game.get_legal_moves())),
            "chosen_move": list(chosen_move),
            "root_value_mean": float(sum(root_values) / len(root_values)) if root_values else None,
            "moves": move_rows,
        }
        with self.debug_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.debug_step += 1

    def _search(self, game):
        legal = game.get_legal_moves()
        mask = np.zeros(180, dtype=np.float32)
        for move in legal:
            idx = REVERSE_LOOKUP[move]
            mask[idx] = 1.0

        self.my_player_idx = game.current_player_idx
        root = MCTSNode(randomize_hidden_bag_for_search(game))
        # 先对 root 做一次 evaluate + expand
        policy_logits, _ = self._evaluate(root.game)
        legal_moves = root.game.get_legal_moves()
        priors = self._get_priors(policy_logits, legal_moves)
        for action, p in priors.items():
            root.add_child(action, prior=p)

        for _ in range(self.n_simulations):
            node = root
            # Selection: 一直走到叶子（没有子节点的节点）
            while node.children and not node.game.is_game_over():
                node = node.best_child(C=self.puct_c)

            # 到叶子了
            if node.game.is_game_over():
                value = self._terminal_value(node)
            else:
                # Expansion + Evaluation
                policy_logits, value = self._evaluate(node.game)
                legal_moves = node.game.get_legal_moves()
                priors = self._get_priors(policy_logits, legal_moves)
                for action, p in priors.items():
                    node.add_child(action, prior=p)

            self._backprop(node, value)

        if not root.children:
            print("not root children")
            legal = game.get_legal_moves()
            move = legal[0]
            pi = np.zeros(self.action_dim, dtype=np.float32)
            idx = REVERSE_LOOKUP[move]
            pi[idx] = 1.0
            return move, pi, mask
        
        move = max(root.children, key=lambda c: c.visits).action
        pi = self._build_pi_from_root(root)

        # # 搜索结束后，打印 root 的子节点统计
        # for child in sorted(root.children, key=lambda c: -c.visits)[:2]:
        #     q = -(child.wins / child.visits) if child.visits > 0 else 0
        #     print(
        #         f"action={child.action} visits={child.visits} wins={child.wins:.2f} Q={q:.3f} prior={child.prior:.3f}")

        return move, pi, mask

    def _build_pi_from_root(self, root):
        pi = np.zeros(self.action_dim, dtype=np.float32)
        for child in root.children:
            idx = REVERSE_LOOKUP[child.action]
            pi[idx] = child.visits
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def _search_multi(self, game):
        legal, mask = self._build_mask(game)

        self.my_player_idx = game.current_player_idx
        n_worlds = max(1, min(self.n_determinizations, self.n_simulations))
        sims_per_world = self.n_simulations // n_worlds
        extra = self.n_simulations % n_worlds

        roots = []
        for world_idx in range(n_worlds):
            sims_this_world = sims_per_world + (1 if world_idx < extra else 0)
            if sims_this_world <= 0:
                continue
            root_game = randomize_hidden_bag_for_search(game)
            roots.append(self._run_single_search(root_game, sims_this_world))

        total_visits, total_wins, total_priors = self._aggregate_roots(roots)

        if not total_visits:
            print("not root children")
            move = legal[0]
            pi = np.zeros(self.action_dim, dtype=np.float32)
            idx = REVERSE_LOOKUP[move]
            pi[idx] = 1.0
            return move, pi, mask

        move = max(total_visits.items(), key=lambda item: item[1])[0]
        pi = self._build_pi_from_visit_dict(total_visits)
        self._write_debug_log(game, move, total_visits, total_wins, total_priors, roots)

        # for action, visits in sorted(total_visits.items(), key=lambda item: -item[1])[:2]:
        #     wins = total_wins.get(action, 0.0)
        #     q = -(wins / visits) if visits > 0 else 0.0
        #     avg_prior = total_priors.get(action, 0.0) / len(roots)
        #     print(
        #         f"action={action} visits={visits} wins={wins:.2f} Q={q:.3f} prior={avg_prior:.3f}")

        return move, pi, mask

    def decide(self, game):
        move, _, _ = self._search_multi(game)
        return move

    def decide_with_info(self, game):
        return self._search_multi(game)

    def _terminal_value(self, node):
        winner = node.game.get_game_result()
        if winner == -1:
            return 0.0  # 平局
        # value 定义为当前节点行动方视角下的终局结果
        current_player = node.game.current_player_idx
        return 1.0 if winner == current_player else -1.0

    def _backprop(self, node, value):
        # value: 神经网络或终局算出的值 (针对 node.game 的当前玩家)
        current = node
        v = value
        while current is not None:
            current.visits += 1
            # current.wins 记录的是：站在该节点视角下，这步棋有多好
            current.wins += v
            # 核心：往父节点走时，视角取反
            # 因为父节点的 wins 是相对于父节点那个玩家的
            v = -v
            current = current.parent
    # 训练流程
    # NN需要训练数据，来源就是MCTS自对战：
    # 1.
    # 用当前MCTS打一局，记录每一步的(obs, mcts_visit_counts, 最终胜负)
    # 2.
    # 用这些数据训练网络：
    # - policy
    # head学习mcts_visit_counts（模仿MCTS的搜索分布）
    # - value
    # head学习最终胜负（1.0或0.0）
    # 3.
    # 用新网络更新MCTS，再打一局，循环

    # 现在整个加NN的工程量其实不小，我想先确认一下你的预期——加NN之后需要 ** 训练 **，训练需要时间和迭代。
    #
    # 整个流程是这样的：
    # ```
    # 初始网络（随机）
    # ↓
    # MCTS用随机网络自对战，收集数据
    # ↓
    # 用数据训练网络
    # ↓
    # 用新网络的MCTS再自对战，收集更好的数据
    # ↓
    # 循环...


# class AzulNet(nn.Module):
#     def __init__(self, obs_dim=562, action_dim=180):
#         super().__init__()
#         self.trunk = nn.Sequential(
#             nn.Linear(obs_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#         )
#         self.policy_head = nn.Linear(256, action_dim)
#         self.value_head = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         feat = self.trunk(x)
#         policy_logits = self.policy_head(feat)
#         value_logit = self.value_head(feat).squeeze(-1)
#         return policy_logits, value_logit


class MCTSAgentGreedy:
    def __init__(self, n_simulations=200, my_player_idx=0):
        self.n_simulations = n_simulations
        self.my_player_idx = my_player_idx
        self.greedy_agent = GreedyAgent()  # 你现有的GreedyAgent

    def _rollout(self, game):
        sim_game = copy.deepcopy(game)
        round_count = 0
        while not sim_game.is_game_over():
            while not sim_game.public_board.is_empty():
                # 换成Greedy决策而不是随机
                move = self.greedy_agent.decide(sim_game)
                sim_game.play_turn(*move)

            sim_game._internal_scoring_flow()
            round_count += 1
            if round_count > 20:
                break

        my_score = sim_game.players[self.my_player_idx].score
        opp_score = sim_game.players[1 - self.my_player_idx].score
        return 1.0 if my_score > opp_score else 0.0

    def decide(self, game):
        # 决策时记录当前是谁在走
        self.my_player_idx = game.current_player_idx

        root = MCTSNode(randomize_hidden_bag_for_search(game))

        for _ in range(self.n_simulations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                # 判断当前节点轮到谁走
                if node.game.current_player_idx == self.my_player_idx:
                    node = node.best_child(C=1.4)  # 我方：选最优
                else:
                    node = node.best_child(C=1.4)  # 对方：暂时也选最优（假设对手也理性）

            # Expansion
            if node.untried_actions:
                action = node.untried_actions.pop()
                node = node.add_child(action)


            # Simulation
            result = self._rollout(node.game)

            # Backpropagation：对手节点的wins要取反
            current = node
            while current is not None:
                current.visits += 1
                if current.player_idx == self.my_player_idx:
                    current.wins += result
                else:
                    current.wins += (1.0 - result)
                current = current.parent

        return max(root.children, key=lambda c: c.visits).action


def collect_data(agent, games=100):
    data = []
    game = AzulGame()

    for i in range(games):
        game.reset()
        episode_data = []
        round_count = 0

        while not game.is_game_over():
            while not game.public_board.is_empty():
                curr_idx = game.current_player_idx
                state = game.get_observation_for_player(curr_idx)
                obs = game.state_to_vector_np(state)

                move, _ = agent.decide(game)
                episode_data.append((obs, curr_idx))
                game.play_turn(*move)

            game._internal_scoring_flow()
            round_count += 1
            if round_count > 20:
                break

        # 胜负标签
        winner = 0 if game.players[0].score > game.players[1].score else 1

        for obs, player_idx in episode_data:
            value = 1.0 if player_idx == winner else 0.0
            data.append((obs, value))

        if (i + 1) % 10 == 0:
            print(f"收集进度: {i + 1}/{games}局, 数据量: {len(data)}")

    return data


if __name__ == "__main__":
    # 用之前能打67%胜率的Greedy rollout版MCTS
    greedy_mcts = MCTSAgentGreedy(n_simulations=200)  # 用greedy rollout的版本
    data = collect_data(greedy_mcts, games=100)
    print(f"收集到{len(data)}条数据")
    with open("search3_greedy_dataset.pkl", "wb") as f:
        pickle.dump(data, f)
    pass
