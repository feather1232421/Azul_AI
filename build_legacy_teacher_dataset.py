import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from config import ACTION_DIM, MAX_PLAYERS
from legacy_transformer_2p import LEGACY_ACTION_DIM, LEGACY_OBS_DIM, load_legacy_transformer
from train_mcts_nn import convert_legacy_obs_to_current_2p, normalize_policy_vector


def _iter_samples(raw_data):
    if not raw_data:
        return
    first = raw_data[0]
    is_episode_grouped = (
        isinstance(first, (list, tuple))
        and len(first) > 0
        and isinstance(first[0], (list, tuple))
    )
    if is_episode_grouped:
        for episode in raw_data:
            for sample in episode:
                yield sample
    else:
        yield from raw_data


def _legacy_net_targets(net, obs_batch, device, policy_temperature):
    obs_tensor = torch.tensor(np.asarray(obs_batch, dtype=np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        policy_logits, value_logits = net(obs_tensor)
        policy_logits = policy_logits / max(float(policy_temperature), 1e-6)
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy().astype(np.float32)
        value = torch.tanh(value_logits).reshape(-1).cpu().numpy().astype(np.float32)
    return policy, value


def _pad_policy_300(policy_180):
    policy_180 = np.asarray(policy_180, dtype=np.float32)
    if policy_180.shape != (LEGACY_ACTION_DIM,):
        raise ValueError(f"Expected legacy policy shape {(LEGACY_ACTION_DIM,)}, got {policy_180.shape}")
    policy_300 = np.zeros(ACTION_DIM, dtype=np.float32)
    policy_300[:LEGACY_ACTION_DIM] = policy_180
    total = float(policy_300.sum())
    if total > 0:
        policy_300 /= total
    return policy_300


def _teacher_value_vec(value):
    z = np.zeros(MAX_PLAYERS, dtype=np.float32)
    z[0] = float(value)
    z[1] = -float(value)
    return z


def _value_mask_2p():
    mask = np.zeros(MAX_PLAYERS, dtype=np.float32)
    mask[:2] = 1.0
    return mask


def _normalize_existing_obs(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape == (LEGACY_OBS_DIM,):
        return convert_legacy_obs_to_current_2p(obs)
    if obs.shape == (1108,):
        return obs
    raise ValueError(f"Unsupported obs shape: {obs.shape}")


def build_teacher_dataset(
    input_paths,
    output_path,
    legacy_model_path,
    max_samples=None,
    batch_size=512,
    policy_source="replay",
    policy_temperature=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_legacy_transformer(legacy_model_path, device)

    output = []
    pending = []

    def flush_pending():
        if not pending:
            return
        legacy_items = [item for item in pending if item.get("legacy_obs") is not None]
        value_by_id = {}
        policy_by_id = {}
        if legacy_items:
            obs_batch = [item["legacy_obs"] for item in legacy_items]
            net_policy_batch, value_batch = _legacy_net_targets(net, obs_batch, device, policy_temperature)
            for item, net_policy, teacher_value in zip(legacy_items, net_policy_batch, value_batch):
                policy_by_id[id(item)] = net_policy
                value_by_id[id(item)] = _teacher_value_vec(teacher_value)

        for item in pending:
            if policy_source == "legacy-net":
                if id(item) not in policy_by_id:
                    raise ValueError("--policy-source legacy-net requires legacy 567 obs samples.")
                pi = _pad_policy_300(policy_by_id[id(item)])
            else:
                pi = normalize_policy_vector(item["replay_pi"], "pi")
            z = value_by_id.get(id(item), item["existing_z"])
            value_mask = item["existing_value_mask"]
            if z is None:
                raise ValueError("No teacher value available for sample.")
            if value_mask is None:
                value_mask = _value_mask_2p()
            output.append((
                item["current_obs"],
                pi,
                np.asarray(z, dtype=np.float32),
                np.asarray(value_mask, dtype=np.float32),
                normalize_policy_vector(item["replay_mask"], "mask"),
            ))
        pending.clear()

    for input_path in input_paths:
        with Path(input_path).open("rb") as f:
            raw_data = pickle.load(f)
        for sample in _iter_samples(raw_data):
            if len(sample) == 4:
                obs, pi, _z, mask = sample
                existing_z = None
                existing_value_mask = None
            elif len(sample) == 5:
                obs, pi, existing_z, existing_value_mask, mask = sample
            else:
                raise ValueError(f"Expected sample len 4 or 5, got {len(sample)} from {input_path}")
            obs = np.asarray(obs, dtype=np.float32)
            legacy_obs = obs if obs.shape == (LEGACY_OBS_DIM,) else None
            pending.append({
                "legacy_obs": legacy_obs,
                "current_obs": _normalize_existing_obs(obs),
                "replay_pi": np.asarray(pi, dtype=np.float32),
                "replay_mask": np.asarray(mask, dtype=np.float32),
                "existing_z": existing_z,
                "existing_value_mask": existing_value_mask,
            })
            if len(pending) >= batch_size:
                flush_pending()
            if max_samples is not None and len(output) + len(pending) >= max_samples:
                flush_pending()
                break
        if max_samples is not None and len(output) >= max_samples:
            break

    flush_pending()
    if max_samples is not None:
        output = output[:max_samples]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(output, f)

    sample = output[0] if output else None
    shapes = [tuple(x.shape) for x in sample] if sample else []
    summary = {
        "output": str(output_path),
        "samples": len(output),
        "policy_source": policy_source,
        "legacy_model": str(legacy_model_path),
        "sample_shapes": shapes,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build a 300-action 2P teacher dataset from legacy 567 replay samples.")
    parser.add_argument("inputs", nargs="+", help="Legacy replay .pkl files with (obs567, pi180, z, mask180) samples.")
    parser.add_argument("--output", default="replays_action300/legacy2p_teacher_smoke.pkl")
    parser.add_argument("--legacy-model", default="models/transformer_champion.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--policy-source", choices=["replay", "legacy-net"], default="replay")
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    args = parser.parse_args()

    summary = build_teacher_dataset(
        input_paths=args.inputs,
        output_path=args.output,
        legacy_model_path=args.legacy_model,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        policy_source=args.policy_source,
        policy_temperature=args.policy_temperature,
    )
    print(summary)


if __name__ == "__main__":
    main()
