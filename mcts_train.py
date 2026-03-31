import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from explore_mtcs import AzulNet


class AzulDataset(Dataset):
    def __init__(self, data):
        self.obs = torch.FloatTensor([d[0] for d in data])
        self.values = torch.FloatTensor([d[1] for d in data])

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.values[idx]


def train_value_net(data_path, save_path, epochs=50):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"加载数据: {len(data)}条")

    # 切分训练集和验证集
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(AzulDataset(train_data), batch_size=256, shuffle=True)
    val_loader = DataLoader(AzulDataset(val_data), batch_size=256, shuffle=False)

    net = AzulNet(obs_dim=142, action_dim=180)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # 训练
        net.train()
        train_loss = 0
        for obs_batch, value_batch in train_loader:
            _, value_pred = net(obs_batch)
            loss = loss_fn(value_pred.squeeze(), value_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for obs_batch, value_batch in val_loader:
                _, value_pred = net(obs_batch)
                loss = loss_fn(value_pred.squeeze(), value_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        # 保存最好的验证集模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)

    print(f"最佳模型在Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    return net


if __name__ == "__main__":
    net = train_value_net("search3_greedy_dataset.pkl", "azul_net_v1.pth", epochs=50)