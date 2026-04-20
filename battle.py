import time
from pathlib import Path

import torch

from logic import AzulGame
from explore_mtcs import MCTSAgent, AzulNet


def play_one_game(agent_0, agent_1):
    game = AzulGame()
    game.reset()
    agents = {0: agent_0, 1: agent_1}

    while not game.is_game_over():
        game.advance_until_next_decision(agents)

    p0 = game.players[0].score
    p1 = game.players[1].score
    return {
        "p0_score": p0,
        "p1_score": p1,
        "margin": p0 - p1,
        "winner": 0 if p0 > p1 else 1 if p1 > p0 else -1,
    }


def battle(agent_0, agent_1, games=50, verbose=True):
    start_time = time.time()
    results = {"p0_win": 0, "p1_win": 0, "draws": 0, "avg_margin": 0.0}
    margins = []

    for game_idx in range(games):
        game_result = play_one_game(agent_0, agent_1)
        margins.append(game_result["margin"])

        if game_result["winner"] == 0:
            results["p0_win"] += 1
        elif game_result["winner"] == 1:
            results["p1_win"] += 1
        else:
            results["draws"] += 1

        if verbose:
            print(
                f"[Game {game_idx + 1:03d}] "
                f"P0 {game_result['p0_score']} vs P1 {game_result['p1_score']} "
                f"| margin={game_result['margin']:+d}"
            )

    total_time = time.time() - start_time
    results["avg_margin"] = sum(margins) / len(margins) if margins else 0.0

    print(f"Final Results: {results}")
    print(f"avg time per game: {total_time / max(games, 1):.3f}s")
    return results


def build_mcts_agent(
    model_path,
    seat,
    device,
    n_simulations=200,
    n_determinizations=4,
    puct_c=1.4,
    prior_temperature=1.0,
    debug_log_path=None,
    debug_label=None,
):
    net = AzulNet(obs_dim=567, action_dim=180)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return MCTSAgent(
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        my_player_idx=seat,
        net=net,
        device=device,
        use_policy=True,
        use_value=True,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        debug_log_path=debug_log_path,
        debug_label=debug_label,
    )


def arena_match(
    model_a_path,
    model_b_path,
    games_per_side=10,
    n_simulations=200,
    n_determinizations=4,
    puct_c=1.4,
    prior_temperature=1.0,
    device=None,
    debug_dir=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a_name = Path(model_a_path).name
    model_b_name = Path(model_b_path).name

    print(f"Arena start: {model_a_name} vs {model_b_name}")
    print(
        f"Device: {device}, games_per_side={games_per_side}, sims={n_simulations}, "
        f"worlds={n_determinizations}, puct_c={puct_c}, prior_T={prior_temperature}"
    )

    debug_dir = Path(debug_dir) if debug_dir else None

    forward_a = build_mcts_agent(
        model_a_path,
        seat=0,
        device=device,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        debug_log_path=debug_dir / f"{Path(model_a_path).stem}_as_p0.jsonl" if debug_dir else None,
        debug_label=f"{Path(model_a_path).stem}_p0",
    )
    forward_b = build_mcts_agent(
        model_b_path,
        seat=1,
        device=device,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        debug_log_path=debug_dir / f"{Path(model_b_path).stem}_as_p1.jsonl" if debug_dir else None,
        debug_label=f"{Path(model_b_path).stem}_p1",
    )
    forward_results = battle(forward_a, forward_b, games=games_per_side, verbose=False)

    reverse_b = build_mcts_agent(
        model_b_path,
        seat=0,
        device=device,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        debug_log_path=debug_dir / f"{Path(model_b_path).stem}_as_p0.jsonl" if debug_dir else None,
        debug_label=f"{Path(model_b_path).stem}_p0",
    )
    reverse_a = build_mcts_agent(
        model_a_path,
        seat=1,
        device=device,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        debug_log_path=debug_dir / f"{Path(model_a_path).stem}_as_p1.jsonl" if debug_dir else None,
        debug_label=f"{Path(model_a_path).stem}_p1",
    )
    reverse_results = battle(reverse_b, reverse_a, games=games_per_side, verbose=False)

    total_games = games_per_side * 2
    model_a_wins = forward_results["p0_win"] + reverse_results["p1_win"]
    model_b_wins = forward_results["p1_win"] + reverse_results["p0_win"]
    draws = forward_results["draws"] + reverse_results["draws"]

    model_a_margin = (
        forward_results["avg_margin"] * games_per_side
        - reverse_results["avg_margin"] * games_per_side
    ) / max(total_games, 1)

    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "games_per_side": games_per_side,
        "total_games": total_games,
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "draws": draws,
        "model_a_win_rate": model_a_wins / max(total_games, 1),
        "model_a_avg_margin": model_a_margin,
    }

    print("Arena Summary:", summary)
    return summary


def sweep_arena(
    model_a_path,
    model_b_path,
    sweep_configs,
    games_per_side=5,
    device=None,
):
    summaries = []
    for config in sweep_configs:
        summary = arena_match(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            games_per_side=games_per_side,
            n_simulations=config.get("n_simulations", 200),
            n_determinizations=config.get("n_determinizations", 4),
            puct_c=config.get("puct_c", 1.4),
            prior_temperature=config.get("prior_temperature", 1.0),
            device=device,
            debug_dir=None,
        )
        summary["config"] = config
        summaries.append(summary)
    print("Sweep Summary:")
    for summary in summaries:
        print(summary)
    return summaries


def promotion_match(
    candidate_path,
    champion_path,
    required_win_rate=0.55,
    games_per_side=20,
    n_simulations=100,
    n_determinizations=4,
    puct_c=1.0,
    prior_temperature=1.0,
    device=None,
):
    summary = arena_match(
        model_a_path=candidate_path,
        model_b_path=champion_path,
        games_per_side=games_per_side,
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        device=device,
        debug_dir=None,
    )
    promoted = summary["model_a_win_rate"] >= required_win_rate
    summary["required_win_rate"] = required_win_rate
    summary["promoted"] = promoted
    print(
        "Promotion Result:",
        {
            "candidate": Path(candidate_path).name,
            "champion": Path(champion_path).name,
            "candidate_win_rate": summary["model_a_win_rate"],
            "required_win_rate": required_win_rate,
            "promoted": promoted,
        },
    )
    return summary


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arena_match(
        model_a_path="azul_net_v4.pt",
        model_b_path="azul_net_v6_last.pt",
        games_per_side=5,
        n_simulations=200,
        n_determinizations=4,
        puct_c=1.0,
        prior_temperature=1.5,
        device=device,
        debug_dir="debug_logs",
    )


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sweep_arena(
#         model_a_path="azul_net_v4.pt",
#         model_b_path="azul_net_v5.pt",
#         games_per_side=3,
#         device=device,
#         sweep_configs=[
#             {"n_simulations": 200, "n_determinizations": 4, "puct_c": 1.4, "prior_temperature": 1.0},
#             {"n_simulations": 200, "n_determinizations": 4, "puct_c": 1.0, "prior_temperature": 1.5},
#             {"n_simulations": 200, "n_determinizations": 8, "puct_c": 1.0, "prior_temperature": 1.5},
#         ],
#     )
