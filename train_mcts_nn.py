import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from azul_net import AzulNet
from dataset import AzulMCTSDataset


def load_raw_data(data_path=None, data_paths=None):
    paths = []
    if data_path is not None:
        paths.append(Path(data_path))
    if data_paths is not None:
        paths.extend(Path(path) for path in data_paths)

    if not paths:
        raise ValueError("At least one data path is required.")

    merged_raw_data = []
    for path in paths:
        with path.open("rb") as f:
            raw_data = pickle.load(f)
        merged_raw_data.extend(raw_data)

    return merged_raw_data, paths


@torch.no_grad()
def evaluate(model, loader, device, value_loss_weight=1.0, loser_policy_weight=0.3):
    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1_match = 0
    total_samples = 0

    for obs, pi, z, mask in loader:
        obs = obs.to(device)
        pi = pi.to(device)
        z = z.to(device)
        mask = mask.to(device)

        policy_logits, value_logit = model(obs)
        value_pred = torch.tanh(value_logit)

        masked_logits = policy_logits.masked_fill(mask == 0, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        per_sample_policy_loss = -(pi * log_probs).sum(dim=-1)
        policy_weights = torch.where(
            z < 0,
            torch.full_like(z, loser_policy_weight),
            torch.ones_like(z),
        )
        policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8)
        value_loss = F.mse_loss(value_pred, z)
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
        and len(first[0]) == 4
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
    value_loss_weight=0.2,
    loser_policy_weight=0.3,
    strict_episode_split=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    raw_data, loaded_paths = load_raw_data(data_path=data_path, data_paths=data_paths)

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

    print("Loaded replay files:")
    for path in loaded_paths:
        print(f" - {path}")
    print(f"Loaded top-level entries: {len(raw_data)}")
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

    model = AzulNet(obs_dim=567, action_dim=180).to(device)
    save_path = Path(save_path)
    last_save_path = save_path.with_name(f"{save_path.stem}_last{save_path.suffix}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)

        if resume_weights_only:
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
            start_epoch = 1
            best_val_loss = float("inf")
            print("Loaded resume weights only.")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resume from epoch {start_epoch}")
        else:
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

        for obs, pi, z, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            obs = obs.to(device)
            pi = pi.to(device)
            z = z.to(device)
            mask = mask.to(device)

            policy_logits, value_logit = model(obs)
            masked_logits = policy_logits.masked_fill(mask == 0, -1e9)
            value_pred = torch.tanh(value_logit)

            log_probs = torch.log_softmax(masked_logits, dim=-1)
            per_sample_policy_loss = -(pi * log_probs).sum(dim=-1)
            policy_weights = torch.where(
                z < 0,
                torch.full_like(z, loser_policy_weight),
                torch.ones_like(z),
            )
            policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_weights.sum().clamp_min(1e-8)
            value_loss = F.mse_loss(value_pred, z)
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )
            print(f" -> Saved best model to {save_path}")

        if epoch % 15 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                last_save_path,
            )
            print(f" -> Saved epoch {epoch} model to {last_save_path}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": start_epoch + epochs - 1,
            "best_val_loss": best_val_loss,
        },
        last_save_path,
    )

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
    train(
        data_path="mcts_dataset_pi_t15_c10_v4_1k.pkl",
        save_path="azul_net_v6.pt",
        resume_path="azul_net_v4.pt",
        batch_size=256,
        lr=2e-4,
        weight_decay=1e-4,
        epochs=15,
        train_ratio=0.9,
        seed=42,
    )
