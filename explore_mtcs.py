import math
import copy
import random
import time
from ai import GreedyAgent
import torch
import torch.nn as nn
from logic import AzulGame
import pickle


class MCTSNode:
    def __init__(self, game, parent=None, action=None, prior=0.0):
        self.game = game  # deepcopy过来的AzulGame实例
        self.parent = parent
        self.action = action  # (从哪拿, 颜色, 放哪里)
        self.player_idx = game.current_player_idx  # add_child之前记录
        self.prior = prior  # 网络给这个动作的先验概率

        self.children = []
        self.untried_actions = self.game.get_legal_moves()  # 你现有的合法动作列表
        random.shuffle(self.untried_actions)  # 打乱顺序，避免偏差

        self.visits = 0
        self.wins = 0.0

    def ucb_score(self, C=1.4):
        if self.visits == 0:
            return float('inf')
        q_value = self.wins / self.visits
        u_value = C * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, C=1.4):
        return max(self.children, key=lambda c: c.ucb_score(C))

    def add_child(self, action):
        # 先记录当前是谁在走
        acting_player = self.game.current_player_idx
        new_game = copy.deepcopy(self.game)
        new_game.play_turn(*action)
        child = MCTSNode(new_game, parent=self, action=action)
        child.player_idx = acting_player  # 记录是谁走了这步
        self.children.append(child)
        return child


class MCTSAgent:
    def __init__(self, n_simulations=200, my_player_idx=0, net=None):
        self.n_simulations = n_simulations
        self.my_player_idx = my_player_idx
        self.net = net if net is not None else AzulNet()
        self.net.eval()

    def _get_obs_tensor(self, game):
        state = game.get_observation_current()
        obs = game.state_to_vector(state)
        return torch.FloatTensor(obs).unsqueeze(0)

    def _evaluate(self, game):
        obs = self._get_obs_tensor(game)
        with torch.no_grad():
            policy_logits, value = self.net(obs)
        return policy_logits.squeeze(0), value.item()

    def decide(self, game):
        self.my_player_idx = game.current_player_idx
        root = MCTSNode(copy.deepcopy(game))

        for _ in range(self.n_simulations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                if node.game.current_player_idx == self.my_player_idx:
                    node = node.best_child(C=1.4)
                else:
                    node = node.best_child(C=1.4)

            # Expansion
            if node.untried_actions:
                action = node.untried_actions.pop()
                node = node.add_child(action)

            # NN估值替代rollout
            _, value = self._evaluate(node.game)

            # Backpropagation
            current = node
            while current is not None:
                current.visits += 1
                if current.player_idx == self.my_player_idx:
                    current.wins += value
                else:
                    current.wins += (1.0 - value)
                current = current.parent

        return max(root.children, key=lambda c: c.visits).action

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


class AzulNet(nn.Module):
    def __init__(self, obs_dim=142, action_dim=180):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        trunk = self.trunk(x)
        value = self.value_head(trunk)
        policy = self.policy_head(trunk)
        return policy, value


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

        root = MCTSNode(copy.deepcopy(game))

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
                obs = game.state_to_vector(state)

                move = agent.decide(game)
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