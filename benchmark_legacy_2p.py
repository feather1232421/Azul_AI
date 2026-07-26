import argparse
from pathlib import Path

import torch

from battle import battle
from config import ACTION_DIM, TRANSFORMER_OBS_DIM
from explore_mtcs import MCTSAgent
from legacy_transformer_2p import LegacyMCTSAgent2P, load_legacy_transformer
from model_utils import load_model


def build_current_agent(model_path, device, sims, worlds, puct_c, prior_temperature):
    net, _, model_type, _ = load_model(
        model_path,
        device=device,
        obs_dim=TRANSFORMER_OBS_DIM,
        action_dim=ACTION_DIM,
        allow_partial_load=True,
    )
    print(f"Loaded current {Path(model_path).name} as {model_type}")
    return MCTSAgent(
        n_simulations=sims,
        n_determinizations=worlds,
        net=net,
        device=device,
        action_dim=ACTION_DIM,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
    )


def build_legacy_agent(model_path, device, sims, worlds, puct_c, prior_temperature):
    net = load_legacy_transformer(model_path, device)
    print(f"Loaded legacy {Path(model_path).name}")
    return LegacyMCTSAgent2P(
        net=net,
        device=device,
        n_simulations=sims,
        n_determinizations=worlds,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the native legacy 2P champion against a current "
            "multiplayer-capable 300-action model on a 2-player Azul table."
        )
    )
    parser.add_argument(
        "--multiplayer-model",
        "--current-model",
        dest="multiplayer_model",
        default="models/transformer_action300_olddata_warmup_plus6.pt",
        help="Current 3P/4P-capable 300-action model to test in a 2P game.",
    )
    parser.add_argument("--legacy-model", default="models/transformer_champion.pt")
    parser.add_argument("--games-per-side", type=int, default=3)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    current_p0 = build_current_agent(args.multiplayer_model, device, args.sims, args.worlds, args.puct_c, args.prior_temperature)
    legacy_p1 = build_legacy_agent(args.legacy_model, device, args.sims, args.worlds, args.puct_c, args.prior_temperature)
    forward = battle(current_p0, legacy_p1, games=args.games_per_side, verbose=False)

    legacy_p0 = build_legacy_agent(args.legacy_model, device, args.sims, args.worlds, args.puct_c, args.prior_temperature)
    current_p1 = build_current_agent(args.multiplayer_model, device, args.sims, args.worlds, args.puct_c, args.prior_temperature)
    reverse = battle(legacy_p0, current_p1, games=args.games_per_side, verbose=False)

    multiplayer_wins = forward["p0_win"] + reverse["p1_win"]
    legacy_wins = forward["p1_win"] + reverse["p0_win"]
    draws = forward["draws"] + reverse["draws"]
    total = args.games_per_side * 2
    multiplayer_margin = (
        forward["avg_margin"] * args.games_per_side
        - reverse["avg_margin"] * args.games_per_side
    ) / max(total, 1)

    print(
        "Legacy Benchmark Summary:",
        {
            "multiplayer_model": Path(args.multiplayer_model).name,
            "legacy_model": Path(args.legacy_model).name,
            "total_games": total,
            "multiplayer_wins": multiplayer_wins,
            "legacy_wins": legacy_wins,
            "draws": draws,
            "multiplayer_win_rate": multiplayer_wins / max(total, 1),
            "multiplayer_avg_margin": multiplayer_margin,
        },
    )


if __name__ == "__main__":
    main()
