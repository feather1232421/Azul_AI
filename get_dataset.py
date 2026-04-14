import pickle
import argparse
from tqdm import tqdm
import torch
from config import *
from ai import GreedyAgent
import numpy as np
from logic import AzulGame
from explore_mtcs import MCTSAgent, AzulNet


def build_action_mask(game):
    mask = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    for move in game.get_legal_moves():
        mask[REVERSE_LOOKUP[move]] = 1.0
    return mask


def build_one_hot_policy(move):
    pi = np.zeros(len(ACTION_LOOKUP), dtype=np.float32)
    pi[REVERSE_LOOKUP[move]] = 1.0
    return pi


def collect_data(agent, games=100):
    data = []

    for game_idx in tqdm(range(games), desc="Collecting self-play data"):
        game = AzulGame()
        game.reset()
        episode = []

        while not game.is_game_over():
            while not game.is_round_over():
                curr_idx = game.current_player_idx
                state = game.get_observation_for_player(curr_idx)
                obs = game.state_to_vector_np(state)

                # 这里的 move, pi, mask 是搜索出来的
                _, pi, mask = agent.decide_with_info(game)

                # --- 核心修改：在这里进行采样 ---
                # 1. 产生动作索引 (0-179)
                # 注意：pi 已经归一化过了，直接用做概率
                move_idx = np.random.choice(len(pi), p=pi)

                # 2. 从索引转回真正的动作元组
                # 假设你有一个 ACTION_LOOKUP 列表存放了所有 180 个动作
                move = ACTION_LOOKUP[move_idx]
                # -----------------------------

                episode.append((np.array(obs, copy=True), np.array(pi, copy=True), curr_idx, np.array(mask, copy=True)))
                game.play_turn(*move)

            # game._internal_scoring_flow()

        if game.players[0].score > game.players[1].score:
            winner = 0
        elif game.players[1].score > game.players[0].score:
            winner = 1
        else:
            winner = 2

        for obs, pi, player_idx, mask in episode:
            if winner == 2:
                z = 0.0
            else:
                z = 1.0 if player_idx == winner else -1.0
            data.append((obs, pi, z, mask))

    return data


def collect_greedy_data(games=100, agent=None):
    data = []
    agent = agent or GreedyAgent()

    for game_idx in tqdm(range(games), desc="Collecting greedy teacher data"):
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

        for obs, pi, player_idx, mask in episode:
            if winner == 2:
                z = 0.0
            else:
                z = 1.0 if player_idx == winner else -1.0
            data.append((obs, pi, z, mask))

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
    args = parser.parse_args()

    if args.mode == "greedy":
        dataset = collect_greedy_data(games=args.games)
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
        dataset = collect_data(agent, games=args.games)
        output_path = args.output or "MCTS_nn_dataset_pi_t15_c10.pkl"

    print("dataset size =", len(dataset))
    print("sample =", dataset[0])

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
