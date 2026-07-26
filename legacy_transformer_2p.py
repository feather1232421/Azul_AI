import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import EMPTY, FIRST_PLAYER, REVERSE_LOOKUP, color_to_onehot
from explore_mtcs import randomize_hidden_bag_for_search, release_search_tree
from logic import AzulGame


LEGACY_ACTION_LOOKUP = [
    (src, color, row)
    for src in ["center", 0, 1, 2, 3, 4]
    for color in range(1, 6)
    for row in range(6)
]
LEGACY_REVERSE_LOOKUP = {move: idx for idx, move in enumerate(LEGACY_ACTION_LOOKUP)}
LEGACY_ACTION_DIM = 180
LEGACY_OBS_DIM = 567


def legacy_state_to_vector_567(game):
    state = game.get_observation_current()
    features = []

    for factory in state["factories"][:5]:
        for tile in factory:
            features.extend(color_to_onehot(tile))

    center_counts = [0.0] * 6
    for tile in state["center"]:
        if 1 <= tile <= 6:
            center_counts[tile - 1] += 1.0
    features.extend(center_counts)

    me = state["me"]
    opp = state["opponents"][0]

    def add_player_board(player):
        for row in player["wall"]:
            features.extend([1.0 if cell else 0.0 for cell in row])
        for line in player["pattern_lines"]:
            padded = line + [0] * (5 - len(line))
            for tile in padded:
                features.extend(color_to_onehot(tile))
        # Preserve the original 567-vector layout used to train transformer_champion.pt.
        # The old transformer slices this region as floor+score, so this apparent
        # score-before-floor order is intentional checkpoint compatibility.
        features.append(float(player["score"]) / 150.0)
        floor = (player["floor"] + [0] * 7)[:7]
        for tile in floor:
            features.extend(color_to_onehot(tile))

    add_player_board(me)
    add_player_board(opp)

    me_first = 1.0 if FIRST_PLAYER in me["floor"] else 0.0
    opp_first = 1.0 if FIRST_PLAYER in opp["floor"] else 0.0
    me_wall_progress = sum(row.count(True) for row in me["wall"]) / 25.0
    opp_wall_progress = sum(row.count(True) for row in opp["wall"]) / 25.0
    score_delta = (float(me["score"]) - float(opp["score"])) / 50.0
    features.extend([me_first, opp_first, me_wall_progress, opp_wall_progress, score_delta])

    vec = np.asarray(features, dtype=np.float32)
    if vec.shape != (LEGACY_OBS_DIM,):
        raise ValueError(f"Legacy obs shape mismatch: {vec.shape}")
    return vec


class LegacyAzulTransformer2P(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=2048):
        super().__init__()
        self.num_factory_tokens = 5
        self.num_tokens = 23

        self.factory_emb = nn.Linear(24, d_model)
        self.center_emb = nn.Linear(6, d_model)
        self.pattern_emb = nn.Linear(30, d_model)
        self.wall_emb = nn.Linear(25, d_model)
        self.floor_emb = nn.Linear(42, d_model)
        self.score_emb = nn.Linear(1, d_model)
        self.global_emb = nn.Linear(5, d_model)
        self.position_emb = nn.Embedding(self.num_tokens, d_model)
        self.token_type_emb = nn.Embedding(11, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, LEGACY_ACTION_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        token_type_ids = [0]
        token_type_ids.extend([1] * self.num_factory_tokens)
        token_type_ids.extend([2])
        token_type_ids.extend([3])
        token_type_ids.extend([4] * 5)
        token_type_ids.extend([5])
        token_type_ids.extend([6])
        token_type_ids.extend([7])
        token_type_ids.extend([8] * 5)
        token_type_ids.extend([9])
        token_type_ids.extend([10])
        self.register_buffer(
            "token_type_ids",
            torch.tensor(token_type_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
        ptr = 0
        factories = x[:, ptr:ptr + 120].view(-1, self.num_factory_tokens, 24)
        ptr += 120
        center = x[:, ptr:ptr + 6].unsqueeze(1)
        ptr += 6

        player_tokens = []
        for _ in range(2):
            wall = x[:, ptr:ptr + 25].unsqueeze(1)
            ptr += 25
            patterns = x[:, ptr:ptr + 150].view(-1, 5, 30)
            ptr += 150
            floor = x[:, ptr:ptr + 42].unsqueeze(1)
            ptr += 42
            score = x[:, ptr:ptr + 1].unsqueeze(1)
            ptr += 1
            player_tokens.extend([
                self.wall_emb(wall),
                self.pattern_emb(patterns),
                self.floor_emb(floor),
                self.score_emb(score),
            ])

        global_feat = x[:, ptr:ptr + 5]
        tokens = [
            self.global_emb(global_feat).unsqueeze(1),
            self.factory_emb(factories),
            self.center_emb(center),
            *player_tokens,
        ]
        combined = torch.cat(tokens, dim=1)
        batch_size = combined.size(0)
        position_ids = torch.arange(self.num_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(batch_size, -1)
        combined = combined + self.position_emb(position_ids) + self.token_type_emb(token_type_ids)
        feat = self.transformer(combined)[:, 0, :]
        return self.policy_head(feat), self.value_head(feat)


def load_legacy_transformer(path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = LegacyAzulTransformer2P().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def softmax_legacy_priors(policy_logits, legal_moves, temperature=1.0):
    logits = []
    moves = []
    for move in legal_moves:
        if move not in LEGACY_REVERSE_LOOKUP:
            continue
        moves.append(move)
        logits.append(float(policy_logits[LEGACY_REVERSE_LOOKUP[move]]) / max(float(temperature), 1e-6))
    max_logit = max(logits)
    exp_logits = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_logits) + 1e-8
    return {move: value / total for move, value in zip(moves, exp_logits)}


class LegacyMCTSNode2P:
    def __init__(self, game, parent=None, action=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.wins = 0.0
        self.visits = 0

    def add_child(self, action, prior):
        child_game = self.game.clone_for_search()
        child_game.play_turn(*action)
        child = LegacyMCTSNode2P(child_game, parent=self, action=action, prior=prior)
        self.children.append(child)
        return child

    def best_child(self, puct_c):
        def score(child):
            q = 0.0 if child.visits == 0 else -(child.wins / child.visits)
            u = puct_c * child.prior * math.sqrt(self.visits + 1e-8) / (1 + child.visits)
            return q + u

        return max(self.children, key=score)


class LegacyMCTSAgent2P:
    def __init__(
        self,
        net,
        device=None,
        n_simulations=200,
        n_determinizations=1,
        puct_c=1.0,
        prior_temperature=1.0,
        use_value=True,
    ):
        self.net = net
        self.device = device or torch.device("cpu")
        self.n_simulations = n_simulations
        self.n_determinizations = max(1, n_determinizations)
        self.puct_c = puct_c
        self.prior_temperature = prior_temperature
        self.use_value = use_value

    def _evaluate(self, game):
        obs = torch.tensor(
            legacy_state_to_vector_567(game),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value_logit = self.net(obs)
            value = float(torch.tanh(value_logit).squeeze().item()) if self.use_value else 0.0
        return policy_logits.squeeze(0), value

    def _terminal_value(self, game):
        winner = game.get_game_result()
        if winner == -1:
            return 0.0
        current_player = game.current_player_idx
        return 1.0 if winner == current_player else -1.0

    def _backprop(self, node, value):
        current = node
        v = value
        while current is not None:
            current.visits += 1
            current.wins += v
            parent = current.parent
            if parent is None:
                break
            if parent.game.current_player_idx != current.game.current_player_idx:
                v = -v
            current = parent

    def _expand(self, node):
        policy_logits, _ = self._evaluate(node.game)
        priors = softmax_legacy_priors(
            policy_logits,
            node.game.get_legal_moves(),
            temperature=self.prior_temperature,
        )
        for action, prior in priors.items():
            node.add_child(action, prior)

    def _run_search(self, root_game, n_simulations):
        root = LegacyMCTSNode2P(root_game)
        self._expand(root)
        for _ in range(n_simulations):
            node = root
            while node.children and not node.game.is_game_over():
                node = node.best_child(self.puct_c)

            if node.game.is_game_over():
                value = self._terminal_value(node.game)
            else:
                policy_logits, scalar_value = self._evaluate(node.game)
                value = scalar_value
                priors = softmax_legacy_priors(
                    policy_logits,
                    node.game.get_legal_moves(),
                    temperature=self.prior_temperature,
                )
                for action, prior in priors.items():
                    node.add_child(action, prior)

            self._backprop(node, value)
        return root

    def decide_with_info(self, game):
        if game.num_players != 2:
            raise ValueError("LegacyMCTSAgent2P only supports 2-player games.")
        legal = game.get_legal_moves()
        mask = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
        for move in legal:
            if move in LEGACY_REVERSE_LOOKUP:
                mask[LEGACY_REVERSE_LOOKUP[move]] = 1.0
        if len(legal) == 1:
            pi = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
            pi[LEGACY_REVERSE_LOOKUP[legal[0]]] = 1.0
            return legal[0], pi, mask

        n_worlds = min(self.n_determinizations, max(1, self.n_simulations))
        sims_per_world = self.n_simulations // n_worlds
        extra = self.n_simulations % n_worlds
        total_visits = {}
        roots = []
        for world_idx in range(n_worlds):
            sims = sims_per_world + (1 if world_idx < extra else 0)
            root = self._run_search(randomize_hidden_bag_for_search(game), sims)
            roots.append(root)
            for child in root.children:
                total_visits[child.action] = total_visits.get(child.action, 0) + child.visits

        if not total_visits:
            move = legal[0]
        else:
            move = max(total_visits.items(), key=lambda item: item[1])[0]
        pi = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
        for action, visits in total_visits.items():
            pi[LEGACY_REVERSE_LOOKUP[action]] = visits
        if pi.sum() > 0:
            pi /= pi.sum()
        for root in roots:
            release_search_tree(root)
        return move, pi, mask

    def decide(self, game):
        move, _, _ = self.decide_with_info(game)
        return move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/transformer_champion.pt")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_legacy_transformer(args.model, device)
    agent = LegacyMCTSAgent2P(net, device=device, n_simulations=args.sims, n_determinizations=args.worlds)
    game = AzulGame(num_players=2)
    move = agent.decide(game)
    print({"model": args.model, "move": move, "obs_dim": LEGACY_OBS_DIM, "action_dim": LEGACY_ACTION_DIM})


if __name__ == "__main__":
    main()
