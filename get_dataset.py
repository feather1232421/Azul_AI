import pickle
from config import *
from ai import GreedyAgent
from search import AzulSearchAgent
from environment import AzulEnv
import numpy as np
from scipy.special import softmax


def collect_regression_data(num_episodes=500, env=None, search_agent=None, temp=3.0):
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


if __name__ == "__main__":
    env = AzulEnv()
    greedy_agent = GreedyAgent()
    search_agent = AzulSearchAgent(
        evaluate_move_fn=greedy_agent.evaluate_move,
        top_k=5,
        verbose=False,
    )
    model = search_agent

    dataset = collect_regression_data(
        num_episodes=600,
        env=env,
        search_agent=search_agent,
    )
    print("dataset size =", len(dataset))
    print("sample =", dataset[0])

    with open("search3_greedy_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
