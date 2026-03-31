import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ScoringDataset(Dataset):
    def __init__(self, X, A, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]


class ScoreModel(nn.Module):
    def __init__(self, obs_dim, action_dim=180, emb_dim=32):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, emb_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action_idx):
        action_feat = self.action_emb(action_idx)
        x = torch.cat([obs, action_feat], dim=1)
        score = self.net(x).squeeze(1)
        return score


def train_score_model(X, A, y, epochs=20, batch_size=512, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ScoringDataset(X, A, y)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ScoreModel(obs_dim=X.shape[1], action_dim=180).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_total = 0

        for obs, action_idx, score in train_loader:
            obs = obs.to(device)
            action_idx = action_idx.to(device)
            score = score.to(device)

            pred = model(obs, action_idx)

            loss = F.mse_loss(pred, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            train_mae += F.l1_loss(pred, score, reduction="sum").item()
            train_total += obs.size(0)

        train_loss /= train_total
        train_mae /= train_total

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_total = 0

        with torch.no_grad():
            for obs, action_idx, score in val_loader:
                obs = obs.to(device)
                action_idx = action_idx.to(device)
                score = score.to(device)

                pred = model(obs, action_idx)

                loss = F.mse_loss(pred, score)

                val_loss += loss.item() * obs.size(0)
                val_mae += F.l1_loss(pred, score, reduction="sum").item()
                val_total += obs.size(0)

        val_loss /= val_total
        val_mae /= val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": X.shape[1],
                "action_dim": 180,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }

        print(
            f"Epoch {epoch:02d} | "
            f"train_mse={train_loss:.4f} train_mae={train_mae:.4f} | "
            f"val_mse={val_loss:.4f} val_mae={val_mae:.4f}"
        )

    return model, best_checkpoint


if __name__ == "__main__":

    set_seed(42)
    with open("greedy_scoring_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    obs_list = []
    action_list = []
    score_list = []

    for obs, action_idx, score in dataset:
        obs_list.append(obs)
        action_list.append(action_idx)
        score_list.append(score)

    X = np.array(obs_list, dtype=np.float32)
    A = np.array(action_list, dtype=np.int64)
    y = np.array(score_list, dtype=np.float32)

    print("X.shape =", X.shape)
    print("A.shape =", A.shape)
    print("y.shape =", y.shape)
    print("score min/max/mean =", y.min(), y.max(), y.mean())

    model, best_checkpoint = train_score_model(
        X, A, y,
        epochs=20,
        batch_size=512,
        lr=1e-3
    )

    torch.save(best_checkpoint, "greedy_score_model_best.pt")
    print(
        f"Best val_mse = {best_checkpoint['best_val_loss']:.4f} "
        f"at epoch {best_checkpoint['epoch']}"
    )