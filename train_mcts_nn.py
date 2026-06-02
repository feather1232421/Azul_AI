import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ACTION_DIM, MAX_PLAYERS, TRANSFORMER_OBS_DIM
from dataset import AzulMCTSDataset
from model_utils import (
    build_checkpoint_payload,
    build_model,
    infer_model_type_from_checkpoint,
    load_checkpoint,
    load_state_dict_partial,
    unwrap_checkpoint_state_dict,
)


LEGACY_OBS_DIM = 567


def convert_legacy_obs_to_current_2p(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape != (LEGACY_OBS_DIM,):
        raise ValueError(f"Expected legacy obs shape {(LEGACY_OBS_DIM,)}, got {obs.shape}")

    new_obs = np.zeros(TRANSFORMER_OBS_DIM, dtype=np.float32)

    # factories: old 5 factories -> new 9 factories (pad remaining 4 factories with zeros)
    new_obs[0:120] = obs[0:120]
    # center
    new_obs[216:222] = obs[120:126]

    me_score = float(obs[343])
    opp_score = float(obs[561])
    me_first = float(obs[562])
    opp_first = float(obs[563])
    opp_delta_vs_me = -float(obs[566])

    ptr = 222

    # me block
    new_obs[ptr + 0] = 1.0
    new_obs[ptr + 1] = me_score
    new_obs[ptr + 2] = 0.0
    new_obs[ptr + 3] = me_first
    ptr += 4
    new_obs[ptr:ptr + 25] = obs[126:151]
    ptr += 25
    new_obs[ptr:ptr + 150] = obs[151:301]
    ptr += 150
    new_obs[ptr:ptr + 42] = obs[301:343]
    ptr += 42

    # opponent block
    new_obs[ptr + 0] = 1.0
    new_obs[ptr + 1] = opp_score
    new_obs[ptr + 2] = opp_delta_vs_me
    new_obs[ptr + 3] = opp_first
    ptr += 4
    new_obs[ptr:ptr + 25] = obs[344:369]
    ptr += 25
    new_obs[ptr:ptr + 150] = obs[369:519]
    ptr += 150
    new_obs[ptr:ptr + 42] = obs[519:561]
    ptr += 42

    # two padded player slots remain zero-filled
    ptr += 2 * (4 + 25 + 150 + 42)

    # global features
    new_obs[ptr + 0] = 2.0 / MAX_PLAYERS
    new_obs[ptr + 1] = 5.0 / 9.0

    return new_obs


def normalize_sample_format(sample):
    if len(sample) == 5:
        obs, pi, z, value_mask, mask = sample
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(pi, dtype=np.float32),
            np.asarray(z, dtype=np.float32),
            np.asarray(value_mask, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
        )

    if len(sample) != 4:
        raise ValueError(f"Unsupported sample format with len={len(sample)}")

    obs, pi, z, mask = sample
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape == (LEGACY_OBS_DIM,):
        obs = convert_legacy_obs_to_current_2p(obs)
    elif obs.shape != (TRANSFORMER_OBS_DIM,):
        raise ValueError(f"Unsupported obs shape: {obs.shape}")

    z_scalar = float(z)
    z_vec = np.zeros(MAX_PLAYERS, dtype=np.float32)
    z_vec[0] = z_scalar
    z_vec[1] = -z_scalar
    value_mask = np.zeros(MAX_PLAYERS, dtype=np.float32)
    value_mask[:2] = 1.0
    return (
        obs,
        np.asarray(pi, dtype=np.float32),
        z_vec,
        value_mask,
        np.asarray(mask, dtype=np.float32),
    )


def normalize_loaded_data(raw_data):
    if not raw_data:
        return raw_data

    first = raw_data[0]
    is_episode_grouped = (
        isinstance(first, (list, tuple))
        and len(first) > 0
        and isinstance(first[0], (list, tuple))
    )

    if is_episode_grouped:
        return [
            [normalize_sample_format(sample) for sample in episode]
            for episode in raw_data
        ]

    return [normalize_sample_format(sample) for sample in raw_data]


def load_raw_data(data_path=None, data_paths=None, repeat_data_paths=None):
    paths = []
    if data_path is not None:
        paths.append(Path(data_path))
    if data_paths is not None:
        paths.extend(Path(path) for path in data_paths)
    if repeat_data_paths is not None:
        for path, repeat in repeat_data_paths:
            paths.extend([Path(path)] * repeat)

    if not paths:
        raise ValueError("At least one data path is required.")

    merged_raw_data = []
    for path in paths:
        with path.open("rb") as f:
            raw_data = pickle.load(f)
        merged_raw_data.extend(normalize_loaded_data(raw_data))

    return merged_raw_data, paths


@torch.no_grad()
def evaluate(model, loader, device, value_loss_weight=1.0, loser_policy_weight=1.0):
    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1_match = 0
    total_samples = 0

    for obs, pi, z, value_mask, mask in loader:
        obs = obs.to(device)
        pi = pi.to(device)
        z = z.to(device)
        value_mask = value_mask.to(device)
        mask = mask.to(device)

        policy_logits, value_logit = model(obs)
        value_pred = torch.tanh(value_logit)

        masked_logits = policy_logits.masked_fill(mask == 0, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        per_sample_policy_loss = -(pi * log_probs).sum(dim=-1)
        policy_target_scalar = z[:, 0]
        policy_weights = torch.where(
            policy_target_scalar < 0,
            torch.full_like(policy_target_scalar, loser_policy_weight),
            torch.ones_like(policy_target_scalar),
        )
        policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8)
        value_sq_error = (value_pred - z) ** 2
        value_loss = (value_sq_error * value_mask).sum() / value_mask.sum().clamp_min(1e-8)
        loss = policy_loss + value_loss_weight * value_loss

        total_loss += loss.item() * obs.size(0)
        total_policy_loss += policy_loss.item() * obs.size(0)
        total_value_loss += value_loss.item() * obs.size(0)

        pred_action = masked_logits.argmax(dim=1)
        target_action = pi.argmax(dim=1)
        total_top1_match += (pred_action == target_action).sum().item()
        total_samples += obs.size(0)

    avg_loss = total_loss / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    top1_match = total_top1_match / total_samples
    return avg_loss, avg_policy_loss, avg_value_loss, top1_match


def split_loaded_data(raw_data, train_ratio=0.9, seed=42):
    if not raw_data:
        return [], [], 0, 0, "empty"

    first = raw_data[0]
    is_episode_grouped = (
        isinstance(first, (list, tuple))
        and len(first) > 0
        and isinstance(first[0], (list, tuple))
        and len(first[0]) == 5
    )

    rng = random.Random(seed)

    if is_episode_grouped:
        episodes = list(raw_data)
        rng.shuffle(episodes)
        if len(episodes) == 1:
            return list(episodes[0]), list(episodes[0]), 1, 1, "episode"

        train_episode_count = int(len(episodes) * train_ratio)
        train_episode_count = min(max(train_episode_count, 1), len(episodes) - 1)
        train_episodes = episodes[:train_episode_count]
        val_episodes = episodes[train_episode_count:]
        train_samples = [sample for episode in train_episodes for sample in episode]
        val_samples = [sample for episode in val_episodes for sample in episode]
        return train_samples, val_samples, len(train_episodes), len(val_episodes), "episode"

    samples = list(raw_data)
    rng.shuffle(samples)
    if len(samples) == 1:
        return samples, samples, 0, 0, "flat"

    train_size = int(len(samples) * train_ratio)
    train_size = min(max(train_size, 1), len(samples) - 1)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    return train_samples, val_samples, 0, 0, "flat"


def train(
    data_path="azul_data.pkl",
    data_paths=None,
    save_path="azul_net_best.pt",
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=15,
    train_ratio=0.9,
    seed=42,
    resume_path=None,
    resume_weights_only=False,
    value_loss_weight=1.0,
    loser_policy_weight=1.0,
    strict_episode_split=False,
    model_type="transformer",
    repeat_data_paths=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    raw_data, loaded_paths = load_raw_data(
        data_path=data_path,
        data_paths=data_paths,
        repeat_data_paths=repeat_data_paths,
    )

    train_samples, val_samples, train_episodes, val_episodes, split_mode = split_loaded_data(
        raw_data,
        train_ratio=train_ratio,
        seed=seed,
    )

    if strict_episode_split and split_mode != "episode":
        raise ValueError(
            "Strict episode split requested, but loaded data is not stored by episode. "
            "Regenerate replay data with by_episode=True."
        )

    print("Loaded data files:")
    for path in loaded_paths:
        print(f" - {path}")
    print(f"Loaded top-level entries: {len(raw_data)}")
    print(
        "Loss weights:",
        {
            "value_loss_weight": value_loss_weight,
            "loser_policy_weight": loser_policy_weight,
        },
    )
    if split_mode == "episode":
        print(f"Episode split: train={train_episodes}, val={val_episodes}")
    elif split_mode == "flat":
        print("Dataset format: flat samples; using random sample split.")
    else:
        print("Dataset format: empty.")

    train_set = AzulMCTSDataset(train_samples)
    val_set = AzulMCTSDataset(val_samples)

    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(model_type=model_type, obs_dim=TRANSFORMER_OBS_DIM, action_dim=ACTION_DIM).to(device)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    last_save_path = save_path.with_name(f"{save_path.stem}_last{save_path.suffix}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, device)
        resume_model_type = infer_model_type_from_checkpoint(ckpt)

        if resume_weights_only:
            if resume_model_type != model_type:
                start_epoch = 1
                best_val_loss = float("inf")
                print(
                    f"Resume skipped: checkpoint model_type={resume_model_type} "
                    f"but current model_type={model_type}."
                )
            else:
                state_dict = unwrap_checkpoint_state_dict(ckpt)
                partial_info = load_state_dict_partial(model, state_dict)
                start_epoch = 1
                best_val_loss = float("inf")
                print(
                    "Loaded resume weights only with partial compatibility:",
                    {
                        "loaded": len(partial_info["loaded_keys"]),
                        "skipped": len(partial_info["skipped"]),
                        "missing": len(partial_info["missing"]),
                    },
                )
        elif isinstance(ckpt, dict) and "model" in ckpt:
            if resume_model_type != model_type:
                raise ValueError(
                    f"Resume checkpoint model_type={resume_model_type} does not match "
                    f"current model_type={model_type}."
                )
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resume from epoch {start_epoch}")
        else:
            if resume_model_type != model_type:
                raise ValueError(
                    f"Resume checkpoint model_type={resume_model_type} does not match "
                    f"current model_type={model_type}."
                )
            model.load_state_dict(ckpt)
            start_epoch = 1
            best_val_loss = float("inf")
            print("Loaded model weights only (no optimizer/epoch info).")
    else:
        start_epoch = 1
        best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_top1_match = 0
        total_samples = 0

        for obs, pi, z, value_mask, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            obs = obs.to(device)
            pi = pi.to(device)
            z = z.to(device)
            value_mask = value_mask.to(device)
            mask = mask.to(device)

            policy_logits, value_logit = model(obs)
            masked_logits = policy_logits.masked_fill(mask == 0, -1e9)
            value_pred = torch.tanh(value_logit)

            log_probs = torch.log_softmax(masked_logits, dim=-1)
            per_sample_policy_loss = -(pi * log_probs).sum(dim=-1)
            policy_target_scalar = z[:, 0]
            policy_weights = torch.where(
                policy_target_scalar < 0,
                torch.full_like(policy_target_scalar, loser_policy_weight),
                torch.ones_like(policy_target_scalar),
            )
            policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8)
            value_sq_error = (value_pred - z) ** 2
            value_loss = (value_sq_error * value_mask).sum() / value_mask.sum().clamp_min(1e-8)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)
            total_policy_loss += policy_loss.item() * obs.size(0)
            total_value_loss += value_loss.item() * obs.size(0)

            pred_action = masked_logits.argmax(dim=1)
            target_action = pi.argmax(dim=1)
            total_top1_match += (pred_action == target_action).sum().item()
            total_samples += obs.size(0)

        train_loss = total_loss / total_samples
        train_policy_loss = total_policy_loss / total_samples
        train_value_loss = total_value_loss / total_samples
        train_top1 = total_top1_match / total_samples

        val_loss, val_policy_loss, val_value_loss, val_top1 = evaluate(
            model,
            val_loader,
            device,
            value_loss_weight=value_loss_weight,
            loser_policy_weight=loser_policy_weight,
        )

        print(
            f"[Epoch {epoch:02d}/{start_epoch + epochs - 1}] "
            f"Train Loss: {train_loss:.4f} "
            f"(P: {train_policy_loss:.4f}, V: {train_value_loss:.4f}) "
            f"| Train Top1: {train_top1:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"(P: {val_policy_loss:.4f}, V: {val_value_loss:.4f}) "
            f"| Val Top1: {val_top1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(build_checkpoint_payload(
                model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"best_val_loss": best_val_loss},
            ), save_path)
            print(f" -> Saved best model to {save_path}")

        if epoch % 15 == 0:
            torch.save(build_checkpoint_payload(
                model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"val_loss": val_loss},
            ), last_save_path)
            print(f" -> Saved epoch {epoch} model to {last_save_path}")

    torch.save(build_checkpoint_payload(
        model,
        optimizer=optimizer,
        epoch=start_epoch + epochs - 1,
        extra={"best_val_loss": best_val_loss},
    ), last_save_path)

    print("Training finished.")
    print("Best val loss:", best_val_loss)
    print("Best model path:", save_path)
    print("Last model path:", last_save_path)
    return {
        "best_val_loss": best_val_loss,
        "best_model_path": str(save_path),
        "last_model_path": str(last_save_path),
        "split_mode": split_mode,
        "loaded_paths": [str(path) for path in loaded_paths],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-paths", type=str, nargs="*", default=None)
    parser.add_argument("--save-path", type=str, default="azul_net_best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--loser-policy-weight", type=float, default=1.0)
    parser.add_argument("--strict-episode-split", action="store_true")
    parser.add_argument("--model-type", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument(
        "--repeat-data-path",
        action="append",
        nargs=2,
        metavar=("PATH", "REPEAT"),
        default=None,
        help="Append a dataset path multiple times, e.g. --repeat-data-path curated.pkl 30.",
    )
    args = parser.parse_args()

    if args.data_path is None and not args.data_paths:
        parser.error("Provide --data-path or --data-paths.")

    train(
        data_path=args.data_path,
        data_paths=args.data_paths,
        save_path=args.save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        seed=args.seed,
        resume_path=args.resume_path,
        resume_weights_only=args.resume_weights_only,
        value_loss_weight=args.value_loss_weight,
        loser_policy_weight=args.loser_policy_weight,
        strict_episode_split=args.strict_episode_split,
        model_type=args.model_type,
        repeat_data_paths=[
            (path, int(repeat))
            for path, repeat in args.repeat_data_path
        ] if args.repeat_data_path else None,
    )
