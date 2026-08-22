import numpy as np
import random
import torch
import torch.nn as nn

from config import ACTION_DIM, MAX_PLAYERS
from explore_mtcs import MCTSAgent, MCTSNode, randomize_hidden_bag_for_search
from logic import AzulGame


class CloneCountingGame:
    def __init__(self, current_player_idx=0, clone_counter=None):
        self.current_player_idx = current_player_idx
        self.clone_counter = clone_counter if clone_counter is not None else [0]
        self.played_actions = []

    def clone_for_search(self):
        self.clone_counter[0] += 1
        clone = CloneCountingGame(self.current_player_idx, self.clone_counter)
        clone.played_actions = list(self.played_actions)
        return clone

    def play_turn(self, *action):
        self.played_actions.append(action)
        self.current_player_idx = (self.current_player_idx + 1) % MAX_PLAYERS


class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, obs):
        batch_size = obs.shape[0]
        self.batch_sizes.append(batch_size)
        return (
            torch.zeros((batch_size, ACTION_DIM), device=obs.device),
            torch.zeros((batch_size, MAX_PLAYERS), device=obs.device),
        )


def test_child_game_is_cloned_only_when_first_materialized():
    root_game = CloneCountingGame()
    root = MCTSNode(root_game)
    child = root.add_child((1, 2, 3), prior=0.5)

    assert child.game is None
    assert root_game.clone_counter[0] == 0

    materialized = child.materialize()

    assert root_game.clone_counter[0] == 1
    assert materialized.played_actions == [(1, 2, 3)]
    assert child.player_idx == 1
    assert child.materialize() is materialized
    assert root_game.clone_counter[0] == 1


def test_determinizations_have_distinct_and_reproducible_bag_rngs():
    game = AzulGame(num_players=4)
    game.reset()
    original_bag = list(game.public_board.bag)
    original_rng_state = game.public_board.rng.getstate()

    first_run = [
        randomize_hidden_bag_for_search(game, random_source)
        for random_source in [random.Random(123)] * 4
    ]
    second_source = random.Random(123)
    second_run = [
        randomize_hidden_bag_for_search(game, second_source)
        for _ in range(4)
    ]

    first_bags = [tuple(world.public_board.bag) for world in first_run]
    second_bags = [tuple(world.public_board.bag) for world in second_run]
    assert len(set(first_bags)) == 4
    assert first_bags == second_bags
    assert game.public_board.bag == original_bag
    assert game.public_board.rng.getstate() == original_rng_state


def test_lazy_nodes_complete_a_real_search_without_mutating_source_game():
    game = AzulGame(num_players=4)
    game.reset()
    before = np.array(game.state_to_vector_np(game.get_observation_current()), copy=True)
    net = ZeroNet()
    agent = MCTSAgent(
        n_simulations=2,
        n_determinizations=2,
        net=net,
        device=torch.device("cpu"),
        root_exploration_fraction=0.0,
    )

    move, pi, mask = agent.decide_with_info(game)

    assert move in game.get_legal_moves()
    assert np.isclose(pi.sum(), 1.0)
    assert mask.sum() == len(game.get_legal_moves())
    assert net.batch_sizes == [2, 2]
    np.testing.assert_array_equal(
        game.state_to_vector_np(game.get_observation_current()),
        before,
    )
