import argparse
from pathlib import Path

from train_mcts_nn import train


def parse_player_mix(value):
    weights = {}
    for item in value.split(","):
        player_count, weight = item.split(":", 1)
        weights[int(player_count)] = float(weight)
    return weights


def build_repeat_data_paths(weighted_paths):
    repeats = []
    for path, repeat in weighted_paths:
        repeat = int(repeat)
        if repeat > 1:
            repeats.append((str(path), repeat - 1))
    return repeats or None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight 2P-heavy mixed training wrapper. "
            "Use legacy teacher data as the anchor and optionally add small 3P/4P datasets."
        )
    )
    parser.add_argument("--teacher-data", default="replays_action300/legacy2p_teacher_all.pkl")
    parser.add_argument("--teacher-repeat", type=int, default=1)
    parser.add_argument("--mixed-data", nargs="*", default=None)
    parser.add_argument("--mixed-repeat", type=int, default=1)
    parser.add_argument("--save-path", default="models/transformer_action300_light_mixed.pt")
    parser.add_argument("--resume-path", default="models/transformer_action300_legacy2p_distill_last.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--loser-policy-weight", type=float, default=1.0)
    parser.add_argument(
        "--player-mix",
        default="2:2,3:1,4:1",
        help="Training sample weights by player count. Default: 2:2,3:1,4:1.",
    )
    parser.add_argument(
        "--resume-full",
        action="store_true",
        help="Resume optimizer/epoch/best-val state. Default loads weights only for a clean fine-tune run.",
    )
    args = parser.parse_args()

    weighted_paths = [(Path(args.teacher_data), args.teacher_repeat)]
    for mixed_path in args.mixed_data or []:
        weighted_paths.append((Path(mixed_path), args.mixed_repeat))

    data_paths = [str(path) for path, _repeat in weighted_paths]
    print("Light mixed training data:")
    for path, repeat in weighted_paths:
        print(f" - {path} x{repeat}")

    train(
        data_path=None,
        data_paths=data_paths,
        save_path=args.save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        seed=args.seed,
        resume_path=args.resume_path,
        resume_weights_only=not args.resume_full,
        value_loss_weight=args.value_loss_weight,
        loser_policy_weight=args.loser_policy_weight,
        strict_episode_split=False,
        model_type="transformer",
        repeat_data_paths=build_repeat_data_paths(weighted_paths),
        player_mix_weights=parse_player_mix(args.player_mix),
    )


if __name__ == "__main__":
    main()
