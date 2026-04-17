import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from ai import GreedyAgent
from config import ACTION_LOOKUP, REVERSE_LOOKUP
from explore_mtcs import AzulNet, MCTSAgent
from logic import AzulGame


def build_action_mask(game):
    mask = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    for move in game.get_legal_moves():
        mask[REVERSE_LOOKUP[move]] = 1.0
    return mask


def build_one_hot_policy(move):
    pi = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    pi[REVERSE_LOOKUP[move]] = 1.0
    return pi


def choose_self_play_move(pi, sample_steps, step_idx):
    if step_idx < sample_steps:
        move_idx = np.random.choice(len(pi), p=pi)
    else:
        move_idx = int(np.argmax(pi))
    return ACTION_LOOKUP[move_idx]


def finalize_episode(episode, winner):
    labeled_episode = []
    for obs, pi, player_idx, mask in episode:
        if winner == 2:
            z = 0.0
        else:
            z = 1.0 if player_idx == winner else -1.0
        labeled_episode.append((obs, pi, z, mask))
    return labeled_episode


def collect_data(agent, games=100, sample_steps=8, by_episode=True):
    data = []

    for _ in tqdm(range(games), desc="Collecting self-play data"):
        game = AzulGame()
        game.reset()
        episode = []
        step_idx = 0

        while not game.is_game_over():
            while not game.is_round_over():
                curr_idx = game.current_player_idx
                state = game.get_observation_for_player(curr_idx)
                obs = game.state_to_vector_np(state)

                _, pi, mask = agent.decide_with_info(game)
                move = choose_self_play_move(pi, sample_steps=sample_steps, step_idx=step_idx)

                episode.append((
                    np.array(obs, copy=True),
                    np.array(pi, copy=True),
                    curr_idx,
                    np.array(mask, copy=True),
                ))
                game.play_turn(*move)
                step_idx += 1

        if game.players[0].score > game.players[1].score:
            winner = 0
        elif game.players[1].score > game.players[0].score:
            winner = 1
        else:
            winner = 2

        labeled_episode = finalize_episode(episode, winner)
        if by_episode:
            data.append(labeled_episode)
        else:
            data.extend(labeled_episode)

    return data


def collect_greedy_data(games=100, agent=None, by_episode=True):
    data = []
    agent = agent or GreedyAgent()

    for _ in tqdm(range(games), desc="Collecting greedy teacher data"):
        game = AzulGame()
        game.reset()
        episode = []

        while not game.is_game_over():
            while not game.is_round_over():
                curr_idx = game.current_player_idx
                state = game.get_observation_for_player(curr_idx)
                obs = game.state_to_vector_np(state)
                mask = build_action_mask(game)

                move = agent.decide(game)
                pi = build_one_hot_policy(move)

                episode.append((
                    np.array(obs, copy=True),
                    np.array(pi, copy=True),
                    curr_idx,
                    np.array(mask, copy=True),
                ))
                game.play_turn(*move)

        if game.players[0].score > game.players[1].score:
            winner = 0
        elif game.players[1].score > game.players[0].score:
            winner = 1
        else:
            winner = 2

        labeled_episode = finalize_episode(episode, winner)
        if by_episode:
            data.append(labeled_episode)
        else:
            data.extend(labeled_episode)

    return data


def get_rank_based_z(rank, num_players):
    if num_players <= 1:
        return 0.0
    return 1.0 - 2.0 * (rank - 1) / (num_players - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mcts", "greedy"], default="greedy")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.5)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--flat", action="store_true")
    args = parser.parse_args()

    if args.mode == "greedy":
        dataset = collect_greedy_data(games=args.games, by_episode=not args.flat)
        output_path = args.output or "greedy_teacher_dataset.pkl"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = AzulNet(obs_dim=567, action_dim=180)
        if args.model is not None:
            ckpt = torch.load(args.model, map_location=device)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            net.load_state_dict(state_dict)
        agent = MCTSAgent(
            n_simulations=args.sims,
            n_determinizations=args.worlds,
            my_player_idx=0,
            net=net,
            device=device,
            action_dim=180,
            puct_c=args.puct_c,
            prior_temperature=args.prior_temperature,
        )
        dataset = collect_data(
            agent,
            games=args.games,
            sample_steps=args.sample_steps,
            by_episode=not args.flat,
        )
        output_path = args.output or "MCTS_nn_dataset_pi_t15_c10.pkl"

    if args.flat:
        total_samples = len(dataset)
        sample = dataset[0] if dataset else None
        episode_count = "n/a"
    else:
        total_samples = sum(len(ep) for ep in dataset)
        sample = dataset[0][0] if dataset and dataset[0] else None
        episode_count = len(dataset)

    print("episodes =", episode_count)
    print("dataset size =", total_samples)
    print("sample =", sample)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
