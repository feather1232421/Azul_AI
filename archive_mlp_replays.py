import argparse
import shutil
from pathlib import Path


MLP_REPLAY_FILES = [
    "selfplay_20260420_192526.pkl",
    "selfplay_20260420_192655.pkl",
    "selfplay_20260420_192905.pkl",
    "selfplay_20260420_200105.pkl",
    "selfplay_20260420_200550.pkl",
    "selfplay_20260421_124350.pkl",
    "selfplay_20260421_135335.pkl",
    "selfplay_20260421_150446.pkl",
    "selfplay_20260421_161622.pkl",
    "selfplay_20260421_172915.pkl",
    "selfplay_20260421_184056.pkl",
    "selfplay_20260421_195225.pkl",
    "selfplay_20260421_210411.pkl",
    "selfplay_20260421_221742.pkl",
    "selfplay_20260421_232827.pkl",
]


def main():
    parser = argparse.ArgumentParser(
        description="Archive known MLP-era self-play replay files out of the active replays directory.",
    )
    parser.add_argument("--source-dir", default="replays")
    parser.add_argument("--archive-dir", default="replays_archive/mlp_legacy")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag, only print the plan.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    for filename in MLP_REPLAY_FILES:
        src = source_dir / filename
        dst = archive_dir / filename
        if src.exists():
            planned.append((src, dst))

    print(f"Found {len(planned)} MLP-era replay files to archive.")
    for src, dst in planned:
        print(f"{src} -> {dst}")

    if not args.apply:
        print("Dry run only. Re-run with --apply after the current transformer self-play finishes.")
        return

    moved = 0
    for src, dst in planned:
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"Moved {moved} replay files to {archive_dir}")


if __name__ == "__main__":
    main()
