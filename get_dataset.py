import pickle
from tqdm import tqdm
import torch
from config import *
from ai import GreedyAgent
from search import AzulSearchAgent
from environment import AzulEnv
import numpy as np
from scipy.special import softmax
from logic import AzulGame
from explore_mtcs import MCTSAgent, AzulNet


def collect_regression_data(num_episodes=500, env=None, search_agent=None, temp=3.0):
    greey_agent = GreedyAgent()
    dataset = []
    game = env.game

    for ep in range(num_episodes):
        game.reset()
        while not game.is_game_over():
            # 轮次结算逻辑
            if game.is_round_over():
                game._internal_scoring_flow()
                if game.is_game_over(): break

            if game.current_player_idx == 0:
                # A. 提取特征
                obs = game.state_to_vector(game.get_observation_for_player(0))

                # B. 获取所有动作的深度搜索评分
                # 这一步是 0.3 秒一局的关键：我们只在玩家 0 决策时搜索
                move_scores_dict = search_agent.evaluate_all_moves(game)

                if not move_scores_dict:
                    break

                # C. 转化为概率分布 (回归目标)
                # 只有被搜过的动作有分，没被搜过的(太烂的)默认给一个很低的分
                full_scores = np.full(180, -999.0, dtype=np.float32)
                for idx, val in move_scores_dict.items():
                    full_scores[idx] = val

                # Softmax 归一化 (Softmax 会自动忽略 -999 那些极小值)
                target_pi = softmax(full_scores / temp)

                # D. 存储样本
                dataset.append((obs.copy(), target_pi))

                # E. 实际走棋：选搜出来的分数最高的那个
                best_action_idx = max(move_scores_dict, key=move_scores_dict.get)
                best_move = ACTION_LOOKUP[best_action_idx]  # 假设你有反查表
                game.play_turn(*best_move)
            else:
                # 对手玩家 1：直接用最快的 Greedy，不搜索
                move = greedy_agent.decide(game)
                game.play_turn(*move)

            if game.is_round_over():
                game._internal_scoring_flow()

    return dataset


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

                move, pi = agent.decide(game)

                episode.append((np.array(obs, copy=True), np.array(pi, copy=True), curr_idx))
                game.play_turn(*move)

            game._internal_scoring_flow()

        if game.players[0].score > game.players[1].score:
            winner = 0
        elif game.players[1].score > game.players[0].score:
            winner = 1
        else:
            winner = 2

        for obs, pi, player_idx in episode:
            if winner == 2:
                z = 0.0
            else:
                z = 1.0 if player_idx == winner else -1.0
            data.append((obs, pi, z))

    return data


def get_rank_based_z(rank, num_players):
    if num_players <= 1:
        return 0.0
    return 1.0 - 2.0 * (rank - 1) / (num_players - 1)


if __name__ == "__main__":
    # env = AzulEnv()
    # greedy_agent = GreedyAgent()
    # search_agent = AzulSearchAgent(
    #     evaluate_move_fn=greedy_agent.evaluate_move,
    #     top_k=5,
    #     verbose=False,
    # )
    # model = search_agent
    #
    # dataset = collect_regression_data(
    #     num_episodes=600,
    #     env=env,
    #     search_agent=search_agent,
    # )
    # print("dataset size =", len(dataset))
    # print("sample =", dataset[0])
    #
    # with open("search3_greedy_dataset.pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AzulNet(obs_dim=562, action_dim=180)
    net.load_state_dict(torch.load("azul_net_best.pt", map_location=device))
    agent = MCTSAgent(n_simulations=200, my_player_idx=0, net=net, device=device, action_dim=180)
    # greedy_agent = GreedyAgent()
    # agent = AzulSearchAgent(
    #     evaluate_move_fn=greedy_agent.evaluate_move,
    #     top_k=5,
    #     verbose=False,
    # )

    dataset = collect_data(agent, games=1)

    print("dataset size =", len(dataset))
    print("sample =", dataset[0])

    with open("MCTS_nn_dataset_pi.pkl", "wb") as f:
        pickle.dump(dataset, f)