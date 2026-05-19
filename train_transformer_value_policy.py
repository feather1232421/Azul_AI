import argparse
from pathlib import Path

from train_mcts_nn import train


DEFAULT_DATA_PATHS = [
    "artifacts/legacy_teacher_data/greedy_teacher_dataset.pkl",
    "artifacts/legacy_mcts_datasets/mcts_dataset_pi_t15_c10_v4_1k.pkl",
]
DEFAULT_RESUME_PATH = "models/transformer_warmstart_d64.pt"
DEFAULT_SAVE_PATH = "models/transformer_policy_value_mix.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Train the transformer policy-value model with the known-good teacher + MCTS mix.",
    )
    parser.add_argument(
        "--data-paths",
        nargs="*",
        default=DEFAULT_DATA_PATHS,
        help="Training datasets to merge. Defaults to greedy teacher + MCTS visit-policy data.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=DEFAULT_SAVE_PATH,
        help="Path for the best checkpoint.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=DEFAULT_RESUME_PATH,
        help="Transformer checkpoint used for warmstart.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.5,
        help="Keep value learning active without letting it dominate policy fitting.",
    )
    parser.add_argument(
        "--loser-policy-weight",
        type=float,
        default=0.5,
        help="Downweight losing positions so noisy policies hurt less.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore warmstart and train a fresh transformer.",
    )
    args = parser.parse_args()

    missing_paths = [path for path in args.data_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing dataset files: {missing_paths}")

    resume_path = None if args.from_scratch else args.resume_path

    train(
        data_path=None,
        data_paths=args.data_paths,
        save_path=args.save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        seed=args.seed,
        resume_path=resume_path,
        resume_weights_only=resume_path is not None,
        value_loss_weight=args.value_loss_weight,
        loser_policy_weight=args.loser_policy_weight,
        model_type="transformer",
    )


if __name__ == "__main__":
    main()
