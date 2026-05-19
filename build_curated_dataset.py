import argparse
import json
import pickle
from pathlib import Path

from curated_cases import CURATED_CASES, build_training_sample


def main():
    parser = argparse.ArgumentParser(
        description="Build a small curated policy-correction dataset from manually reviewed Azul positions.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/curated_positions/curated_policy_cases.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--best-prob",
        type=float,
        default=1.0,
        help="Probability mass assigned to the curated best move.",
    )
    parser.add_argument(
        "--by-episode",
        action="store_true",
        help="Store each curated case as a single-sample episode for compatibility with episode splits.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    manifest_rows = []
    for case in CURATED_CASES:
        sample = build_training_sample(case, best_prob=args.best_prob)
        samples.append([sample] if args.by_episode else sample)
        manifest_rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "best_move": list(case["best_move"]),
                "z": float(case.get("z", 0.0)),
                "description": case["description"],
            }
        )

    with output_path.open("wb") as f:
        pickle.dump(samples, f)

    manifest_path = output_path.with_suffix(".jsonl")
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved curated dataset to {output_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"num_cases={len(CURATED_CASES)} by_episode={args.by_episode} best_prob={args.best_prob}")


if __name__ == "__main__":
    main()
