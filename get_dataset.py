import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from ai import GreedyAgent
from config import ACTION_LOOKUP, REVERSE_LOOKUP
from explore_mtcs import AzulNet, MCTSAgent
from logic import AzulGame

DEFAULT_TEMPERATURE_SCHEDULE = (
    (0, 1.25),
    (12, 0.80),
    (24, 0.35),
    (40, 0.15),
)


def build_action_mask(game):
    mask = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    for move in game.get_legal_moves():
        mask[REVERSE_LOOKUP[move]] = 1.0
    return mask


def build_one_hot_policy(move):
    pi = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    pi[REVERSE_LOOKUP[move]] = 1.0
    return pi


def parse_temperature_schedule(text):
    if not text:
        return list(DEFAULT_TEMPERATURE_SCHEDULE)

    schedule = []
    for item in text.split(","):
        start_text, temp_text = item.split(":")
        schedule.append((int(start_text.strip()), float(temp_text.strip())))
    schedule.sort(key=lambda item: item[0])
    return schedule


def get_temperature_for_step(step_idx, temperature_schedule):
    schedule = temperature_schedule or DEFAULT_TEMPERATURE_SCHEDULE
    temperature = schedule[0][1]
    for start_step, scheduled_temp in schedule:
        if step_idx >= start_step:
            temperature = scheduled_temp
        else:
            break
    return temperature


def temper_policy(pi, temperature):
    temperature = max(float(temperature), 1e-6)
    if temperature <= 1e-3:
        sharpened = np.zeros_like(pi)
        sharpened[int(np.argmax(pi))] = 1.0
        return sharpened

    adjusted = np.array(pi, copy=True, dtype=np.float64)
    positive = adjusted > 0
    adjusted[positive] = np.power(adjusted[positive], 1.0 / temperature)
    total = adjusted.sum()
    if total <= 0:
        fallback = np.zeros_like(pi)
        fallback[int(np.argmax(pi))] = 1.0
        return fallback
    adjusted /= total
    return adjusted.astype(np.float32)


def choose_self_play_move(pi, step_idx, temperature_schedule=None):
    temperature = get_temperature_for_step(step_idx, temperature_schedule)
    tempered_pi = temper_policy(pi, temperature)
    move_idx = np.random.choice(len(tempered_pi), p=tempered_pi)
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


def collect_data(agent, games=100, by_episode=True, temperature_schedule=None):
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
                move = choose_self_play_move(
                    pi,
                    step_idx=step_idx,
                    temperature_schedule=temperature_schedule,
                )

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
    parser.add_argument(
        "--temperature-schedule",
        type=str,
        default="0:1.25,12:0.8,24:0.35,40:0.15",
        help="Comma-separated step:temperature pairs, e.g. 0:1.25,12:0.8,24:0.35",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--flat", action="store_true")
    args = parser.parse_args()
    temperature_schedule = parse_temperature_schedule(args.temperature_schedule)

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
            root_dirichlet_alpha=args.dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
        )
        dataset = collect_data(
            agent,
            games=args.games,
            by_episode=not args.flat,
            temperature_schedule=temperature_schedule,
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
