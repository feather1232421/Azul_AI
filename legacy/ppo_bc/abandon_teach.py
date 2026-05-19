import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GreedyDataset(Dataset):
    def __init__(self, X, M, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=180):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_bc_model(X, M, y, epochs=30, batch_size=256, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = GreedyDataset(X, M, y)

    n_total = len(full_dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = BCPolicy(obs_dim=X.shape[1], action_dim=M.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, mask, action in train_loader:
            obs = obs.to(device)
            mask = mask.to(device)
            action = action.to(device)

            logits = model(obs)
            masked_logits = logits.masked_fill(~mask, -1e9)

            loss = F.cross_entropy(masked_logits, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            pred = masked_logits.argmax(dim=1)
            train_correct += (pred == action).sum().item()
            train_total += obs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_top3_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs, mask, action in val_loader:
                obs = obs.to(device)
                mask = mask.to(device)
                action = action.to(device)

                logits = model(obs)
                masked_logits = logits.masked_fill(~mask, -1e9)
                loss = F.cross_entropy(masked_logits, action)

                val_loss += loss.item() * obs.size(0)

                pred = masked_logits.argmax(dim=1)
                val_correct += (pred == action).sum().item()

                top3 = masked_logits.topk(k=3, dim=1).indices
                val_top3_correct += (top3 == action.unsqueeze(1)).any(dim=1).sum().item()

                val_total += obs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_top3_acc = val_top3_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": X.shape[1],
                "action_dim": M.shape[1],
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top3_acc={val_top3_acc:.4f}"
        )

    return model, best_checkpoint


if __name__ == "__main__":
    set_seed(42)
    with open("greedy_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    obs_list = []
    mask_list = []
    action_list = []

    for obs, mask, action in dataset:
        obs_list.append(obs)
        mask_list.append(mask)
        action_list.append(action)

    X = np.array(obs_list, dtype=np.float32)
    M = np.array(mask_list, dtype=bool)
    y = np.array(action_list, dtype=np.int64)

    print("X.shape =", X.shape)
    print("M.shape =", M.shape)
    print("y.shape =", y.shape)

    model, best_checkpoint = train_bc_model(X, M, y, epochs=30, batch_size=256, lr=1e-3)

    torch.save(best_checkpoint, "bc_greedy_policy_best.pt")
    print(f"Best val_acc = {best_checkpoint['best_val_acc']:.4f} at epoch {best_checkpoint['epoch']}")