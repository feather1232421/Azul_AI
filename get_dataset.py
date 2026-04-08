import pickle
from tqdm import tqdm
import torch
from config import *
from ai import GreedyAgent
import numpy as np
from scipy.special import softmax
from logic import AzulGame
from explore_mtcs import MCTSAgent, AzulNet


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
                obs = game.state_to_vector_new(state)

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


def get_rank_based_z(rank, num_players):
    if num_players <= 1:
        return 0.0
    return 1.0 - 2.0 * (rank - 1) / (num_players - 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AzulNet(obs_dim=562, action_dim=180)
    net.load_state_dict(torch.load("azul_net_best.pt", map_location=device))
    agent = MCTSAgent(n_simulations=200, my_player_idx=0, net=net, device=device, action_dim=180)

    dataset = collect_data(agent, games=100)

    print("dataset size =", len(dataset))
    print("sample =", dataset[0])

    with open("MCTS_nn_dataset_pi.pkl", "wb") as f:
        pickle.dump(dataset, f)
