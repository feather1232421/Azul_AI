import argparse

from config import ACTION_DIM, TRANSFORMER_OBS_DIM
from curated_cases import CURATED_CASES, build_game_from_case
from explore_mtcs import MCTSAgent
from model_utils import load_model
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a model solves a small set of curated Azul positions.",
    )
    parser.add_argument("--model", default="models/transformer_multiplayer_base.pt")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, resolved_model_type, _ = load_model(
        args.model,
        device=device,
        obs_dim=TRANSFORMER_OBS_DIM,
        action_dim=ACTION_DIM,
        allow_partial_load=True,
    )
    print(f"Loaded {args.model} as {resolved_model_type} on {device}")

    agent = MCTSAgent(
        n_simulations=args.sims,
        n_determinizations=args.worlds,
        my_player_idx=0,
        net=net,
        device=device,
        action_dim=ACTION_DIM,
        puct_c=args.puct_c,
        prior_temperature=args.prior_temperature,
        root_dirichlet_alpha=0.0,
        root_exploration_fraction=0.0,
    )

    passed = 0
    for case in CURATED_CASES:
        game = build_game_from_case(case)
        move, _, _ = agent.decide_with_info(game)
        ok = move == case["best_move"]
        print(
            {
                "id": case["id"],
                "chosen_move": move,
                "expected_move": case["best_move"],
                "pass": ok,
            }
        )
        passed += int(ok)

    print(f"passed={passed}/{len(CURATED_CASES)}")


if __name__ == "__main__":
    main()
