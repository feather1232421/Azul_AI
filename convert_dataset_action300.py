import argparse
import pickle
from pathlib import Path

from train_mcts_nn import normalize_loaded_data


def count_samples(data):
    if not data:
        return 0
    first = data[0]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        return sum(len(episode) for episode in data)
    return len(data)


def sample_shapes(data):
    if not data:
        return {}
    sample = data[0][0] if isinstance(data[0], (list, tuple)) and data[0] and isinstance(data[0][0], (list, tuple)) else data[0]
    obs, pi, z, value_mask, mask = sample
    return {
        "obs": tuple(obs.shape),
        "pi": tuple(pi.shape),
        "z": tuple(z.shape),
        "value_mask": tuple(value_mask.shape),
        "mask": tuple(mask.shape),
    }


def convert_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("rb") as f:
        raw_data = pickle.load(f)

    converted = normalize_loaded_data(raw_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(converted, f)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "top_level_entries": len(converted),
        "samples": count_samples(converted),
        "sample_shapes": sample_shapes(converted),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert legacy Azul datasets to the current 300-action format.")
    parser.add_argument("inputs", nargs="+", help="Input .pkl dataset files.")
    parser.add_argument("--output-dir", default="replays_action300", help="Directory for converted datasets.")
    parser.add_argument("--suffix", default="_action300", help="Suffix added before .pkl.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for input_text in args.inputs:
        input_path = Path(input_text)
        output_path = output_dir / f"{input_path.stem}{args.suffix}{input_path.suffix}"
        summary = convert_file(input_path, output_path)
        print(summary)


if __name__ == "__main__":
    main()
