from sb3_contrib import MaskablePPO
from environment import AzulEnv
from ai import GreedyAgent, RandomAgent, PPOAgent, BCAgent, ScoreAgent
from logic import AzulGame
from search import AzulSearchAgent
from config import *
import time
from explore_mtcs import MCTSAgent,AzulNet,MCTSAgentGreedy
import torch


def battle(agent_0, agent_1, games=50):
    game = AzulGame()
    start_time = time.time()  # ⏱️ 开始计时
    results = {"p0_win": 0, "p1_win": 0, "draws": 0}

    for i in range(games):
        game.reset()
        # 这里的 agents 字典是给 advance 用的
        agents = {0: agent_0, 1: agent_1}

        # 🌟 真正的“魔法时刻”：循环直到终点
        while not game.is_game_over():
            # 这个函数会根据当前的 game.current_player_idx
            # 去 agents 字典里找对应的 agent，调用它的 decide()
            # 然后执行 play_turn，如此往复，直到：
            # 1. 游戏结束
            # 2. 或者 agents 字典里没提供当前玩家（这里我们全提供了，所以不会停）
            game.advance_until_next_decision(agents)

        # 结算
        p0, p1 = game.players[0].score, game.players[1].score
        if p0 > p1:
            results["p0_win"] += 1
        elif p1 > p0:
            results["p1_win"] += 1
        else:
            results["draws"] += 1

        print(f"P0  {p0} vs P1  {p1}")

    end_time = time.time()  # ⏱️ 结束计时
    total_time = end_time - start_time
    print(f"Final Results: {results}")
    print(f"avg time: {total_time/games}")


if __name__ == "__main__":
    # 1. 创建环境
    env = AzulEnv()

    # 2. 🌟 关键：清空内部对手池
    # 这样 env.step 内部的 advance_until_next_decision 会直接 break

    # 3. 如果你的 reset 也有快进，也要确保它不跑
    # 或者直接手动调用 game.reset()
    # model_1 = MaskablePPO.load("azul_ppo_v6.zip")
    # model_0 = MaskablePPO.load("azul_ppo_v7.zip")
    # model_0 = PPOAgent("azul_ppo_v8.zip")
    # model_0 = BCAgent("bc_greedy_policy_best.pt")
    # model_0 = PPOAgent("azul_ppo_bc_finetuned")
    # model_0 = ScoreAgent("greedy_score_model_best.pt",0)
    # model_0 = BCAgent("bc_distill_best.pt", target_player_idx=0)
    # model_0 = PPOAgent("azul_ppo_v8.zip")
    # model_0 = PPOAgent("Azul_V9_Final")
    # model_0 = MCTSAgent(n_simulations=200, my_player_idx=0)
    # model_1 = GreedyAgent()
    # model_0 = GreedyAgent()
    #
    # ppoagent = PPOAgent("azul_ppo_v8.zip")
    # greedy_agent = GreedyAgent()
    # search_agent = AzulSearchAgent(
    #     evaluate_move_fn=greedy_agent.evaluate_move,
    #     top_k=5,
    #     verbose=False,
    # )
    # model_0 = search_agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = AzulNet(obs_dim=562, action_dim=180)
    ckpt = torch.load("azul_net_v2.pt", map_location=device)
    net.load_state_dict(ckpt["model"])
    # net.load_state_dict(torch.load("azul_net_best.pt", map_location=device))
    net.to(device)
    net.eval()
    model_0 = MCTSAgent(n_simulations=500,my_player_idx=0,net=net,device=device)
    # model_1 = PPOAgent("azul_ppo_v10.zip")
    # net = AzulNet()
    # model_0 = MCTSAgent(n_simulations=200, net=net)
    # model_0 = MCTSAgentGreedy(n_simulations=200)
    model_1 = GreedyAgent()
    battle(model_0, model_1, games=10)


