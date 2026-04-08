import pickle
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from azul_net import AzulNet
from dataset import AzulMCTSDataset
from tqdm import tqdm
# =========================
# 1. 网络
# =========================

# class AzulNet(nn.Module):
#     def __init__(self, obs_dim=562, action_dim=180):
#         super().__init__()
#         self.trunk = nn.Sequential(
#             nn.Linear(obs_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#         )
#         self.policy_head = nn.Linear(256, action_dim)
#         self.value_head = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         feat = self.trunk(x)
#         policy_logits = self.policy_head(feat)          # [B, 180]
#         value_logit = self.value_head(feat).squeeze(-1) # [B]
#         return policy_logits, value_logit


# =========================
# 2. Dataset
# =========================
# class AzulDataset(Dataset):
#     def __init__(self, data_list):
#         # data_list: list of (obs, action, value)
#         # obs: np.ndarray, shape=(562,)
#         # action: int
#         # value: float (0.0 or 1.0)
#         self.data = data_list
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         obs, action, value = self.data[idx]
#
#         obs = torch.tensor(obs, dtype=torch.float32)
#         action = torch.tensor(action, dtype=torch.long)
#         value = torch.tensor(value, dtype=torch.float32)
#
#         return obs, action, value


# =========================
# 3. 评估函数
# =========================


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1_match = 0
    total_samples = 0

    for obs, pi, z, mask in loader:
        obs = obs.to(device)   # [B, 562]
        pi = pi.to(device)     # [B, 180]
        z = z.to(device)       # [B]
        mask = mask.to(device)

        policy_logits, value_logit = model(obs)   # [B,180], [B]
        value_pred = torch.tanh(value_logit)      # [B]

        # 1. 把非法动作位置变成极小值
        masked_logits = policy_logits.masked_fill(mask == 0, -1e9)

        # policy loss: target distribution
        # 2. 再做 softmax
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        # 3. 正常算 CE
        # policy loss: target is a distribution
        policy_loss = -(pi * log_probs).sum(dim=-1).mean()

        # value loss: z in {-1,0,1}
        value_loss = F.mse_loss(value_pred, z)

        loss = policy_loss + value_loss

        total_loss += loss.item() * obs.size(0)
        total_policy_loss += policy_loss.item() * obs.size(0)
        total_value_loss += value_loss.item() * obs.size(0)

        # 一个“近似可看”的指标：argmax 是否和 pi 的 argmax 一致
        pred_action = policy_logits.argmax(dim=1)
        target_action = pi.argmax(dim=1)
        total_top1_match += (pred_action == target_action).sum().item()
        total_samples += obs.size(0)

    avg_loss = total_loss / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    top1_match = total_top1_match / total_samples

    return avg_loss, avg_policy_loss, avg_value_loss, top1_match


# =========================
# 4. 训练主函数
# =========================
def train(
    data_path="azul_data.pkl",
    save_path="azul_net_best.pt",
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=15,
    train_ratio=0.9,
    seed=42,
    resume_path=None,   # 👈 新增
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(data_path, "rb") as f:
        data_list = pickle.load(f)

    print(f"Loaded data size: {len(data_list)}")

    dataset = AzulMCTSDataset(data_list)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    # DataLoader喽
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 建模型
    model = AzulNet(obs_dim=567, action_dim=180).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 加载权重（如有）
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resume from epoch {start_epoch}")
        else:
            # 兼容旧版：直接就是 state_dict
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

        # for obs, pi, z in train_loader:
        for obs, pi, z, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):

            obs = obs.to(device)   # [B, 562]
            pi = pi.to(device)     # [B, 180]
            z = z.to(device)       # [B]
            mask = mask.to(device) # [B, 180]

            policy_logits, value_logit = model(obs)   # [B,180], [B]
            # 1. 把非法动作位置变成极小值
            masked_logits = policy_logits.masked_fill(mask == 0, -1e9)
            value_pred = torch.tanh(value_logit)      # [B]

            # policy loss: target distribution
            # 2. 再做 softmax
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            # 3. 正常算 CE
            policy_loss = -(pi * log_probs).sum(dim=-1).mean()

            # value loss: z in {-1,0,1}
            value_loss = F.mse_loss(value_pred, z)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)
            total_policy_loss += policy_loss.item() * obs.size(0)
            total_value_loss += value_loss.item() * obs.size(0)

            pred_action = policy_logits.argmax(dim=1)
            kl = (pi * (torch.log(pi + 1e-8) - log_probs)).sum(dim=-1).mean()
            target_action = pi.argmax(dim=1)
            total_top1_match += (pred_action == target_action).sum().item()
            total_samples += obs.size(0)

        train_loss = total_loss / total_samples
        train_policy_loss = total_policy_loss / total_samples
        train_value_loss = total_value_loss / total_samples
        train_top1 = total_top1_match / total_samples

        val_loss, val_policy_loss, val_value_loss, val_top1 = evaluate(
            model, val_loader, device
        )

        print(
            f"[Epoch {epoch:02d}/{start_epoch+epochs-1}] "
            f"Train Loss: {train_loss:.4f} "
            f"(P: {train_policy_loss:.4f}, V: {train_value_loss:.4f}) "
            f"| Train Top1: {train_top1:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"(P: {val_policy_loss:.4f}, V: {val_value_loss:.4f}) "
            f"| Val Top1: {val_top1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }, save_path)
            print(f" -> Saved best model to {save_path}")

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    train(
        data_path="MCTS_nn_dataset_pi.pkl",   # 改成你的数据文件名
        save_path="azul_net_v3.pt",
        resume_path="azul_net_best.pt",
        batch_size=256,
        lr=5e-4,
        weight_decay=1e-4,
        epochs=15,
        train_ratio=0.9,
        seed=42,
    )