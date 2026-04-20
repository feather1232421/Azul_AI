import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import torch

from battle import promotion_match
from explore_mtcs import AzulNet, MCTSAgent
from get_dataset import collect_data, parse_temperature_schedule
from train_mcts_nn import train


def load_net(model_path, device):
    net = AzulNet(obs_dim=567, action_dim=180)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


def collect_self_play(
    champion_path,
    output_path,
    games,
    device,
    n_simulations,
    n_determinizations,
    puct_c,
    prior_temperature,
    dirichlet_alpha,
    root_exploration_fraction,
    temperature_schedule,
):
    net = load_net(champion_path, device)
    agent = MCTSAgent(
        n_simulations=n_simulations,
        n_determinizations=n_determinizations,
        my_player_idx=0,
        net=net,
        device=device,
        action_dim=180,
        puct_c=puct_c,
        prior_temperature=prior_temperature,
        root_dirichlet_alpha=dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )
    dataset = collect_data(
        agent,
        games=games,
        by_episode=True,
        temperature_schedule=temperature_schedule,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    sample_count = sum(len(episode) for episode in dataset)
    print(f"Saved self-play replay to {output_path}")
    print(f"Collected episodes={len(dataset)}, samples={sample_count}")
    return output_path


def select_replay_files(replay_dir, replay_window):
    replay_files = sorted(
        replay_dir.glob("selfplay_*.pkl"),
        key=lambda path: path.stat().st_mtime,
    )
    if replay_window is not None and replay_window > 0:
        replay_files = replay_files[-replay_window:]
    return replay_files


def archive_and_promote(champion_path, candidate_path, archive_dir, iteration_tag):
    archive_dir.mkdir(parents=True, exist_ok=True)
    champion_path = Path(champion_path)
    archived_path = archive_dir / f"{champion_path.stem}_{iteration_tag}{champion_path.suffix}"
    shutil.copy2(champion_path, archived_path)
    shutil.copy2(candidate_path, champion_path)
    return archived_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", type=str, default="azul_net_best.pt")
    parser.add_argument("--replay-dir", type=str, default="replays")
    parser.add_argument("--archive-dir", type=str, default="champion_archive")
    parser.add_argument("--candidate-dir", type=str, default="models")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--replay-window", type=int, default=5)
    parser.add_argument("--selfplay-sims", type=int, default=80)
    parser.add_argument("--selfplay-worlds", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--arena-games-per-side", type=int, default=20)
    parser.add_argument("--arena-sims", type=int, default=100)
    parser.add_argument("--arena-worlds", type=int, default=4)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--temperature-schedule",
        type=str,
        default="0:1.25,12:0.8,24:0.35,40:0.15",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-promote", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    champion_path = Path(args.champion)
    replay_dir = Path(args.replay_dir)
    archive_dir = Path(args.archive_dir)
    candidate_dir = Path(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    iteration_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = replay_dir / f"selfplay_{iteration_tag}.pkl"
    candidate_path = candidate_dir / f"candidate_{iteration_tag}.pt"
    temperature_schedule = parse_temperature_schedule(args.temperature_schedule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Champion: {champion_path}")
    print(f"Device: {device}")
    print(f"Iteration tag: {iteration_tag}")

    collect_self_play(
        champion_path=champion_path,
        output_path=replay_path,
        games=args.games,
        device=device,
        n_simulations=args.selfplay_sims,
        n_determinizations=args.selfplay_worlds,
        puct_c=args.puct_c,
        prior_temperature=args.prior_temperature,
        dirichlet_alpha=args.dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        temperature_schedule=temperature_schedule,
    )

    replay_files = select_replay_files(replay_dir, args.replay_window)
    print("Training from replay files:")
    for replay_file in replay_files:
        print(f" - {replay_file}")

    train_summary = train(
        data_path=None,
        data_paths=[str(path) for path in replay_files],
        save_path=str(candidate_path),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.train_epochs,
        train_ratio=args.train_ratio,
        seed=args.seed,
        resume_path=str(champion_path),
        resume_weights_only=True,
        strict_episode_split=True,
    )

    promotion_summary = promotion_match(
        candidate_path=train_summary["best_model_path"],
        champion_path=str(champion_path),
        required_win_rate=args.promotion_threshold,
        games_per_side=args.arena_games_per_side,
        n_simulations=args.arena_sims,
        n_determinizations=args.arena_worlds,
        puct_c=args.puct_c,
        prior_temperature=args.prior_temperature,
        device=device,
    )

    if args.no_promote:
        print("Promotion disabled by --no-promote. Champion unchanged.")
    elif not args.allow_promote:
        print("Promotion disabled. Pass --allow-promote to update champion.")
    elif promotion_summary["promoted"]:
        archived_path = archive_and_promote(
            champion_path=champion_path,
            candidate_path=train_summary["best_model_path"],
            archive_dir=archive_dir,
            iteration_tag=iteration_tag,
        )
        print(f"Promoted candidate to champion: {champion_path}")
        print(f"Archived previous champion to: {archived_path}")
    else:
        print("Candidate did not beat champion. Champion unchanged.")


if __name__ == "__main__":
    main()
