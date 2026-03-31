import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import numpy as np


# 定义模型 (保持和 PPO 结构对齐)
class DistillModel(nn.Module):
    def __init__(self, obs_dim=142, action_dim=180):  # 根据你的具体维度修改
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 1. 加载数据
    with open("search3_greedy_dataset.pkl", "rb") as f:
        raw_data = pickle.load(f)

    obs_list = [item[0] for item in raw_data]
    pi_list = [item[1] for item in raw_data]

    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(pi_list), dtype=torch.float32)

    # 2. 划分训练集和验证集 (90% / 10%)
    dataset = TensorDataset(X, Y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_db, val_db = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_db, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=256)

    # 定义模型 (保持和 PPO 结构对齐)
    model = DistillModel(obs_dim=X.shape[1], action_dim=180)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # KLDivLoss 注意事项: 输入应为 log_softmax
    criterion = nn.KLDivLoss(reduction='batchmean')

    # 4. 训练循环
    epochs = 50
    best_val_loss = float('inf')  # 初始化为无穷大

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            # 🌟 关键：对输出做 log_softmax，对目标直接使用概率分布
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证环节
        model.eval()
        current_val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                lp = torch.log_softmax(model(bx), dim=1)
                current_val_loss += criterion(lp, by).item()

        avg_val_loss = current_val_loss / len(val_loader)

        # 🌟 核心逻辑：只在进步时存档
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "bc_distill_best.pt")
            print(f"✨ 发现更好的模型！Val Loss: {avg_val_loss:.6f}，已保存。")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
    # 5. 保存模型权重 (后续给 PPO 加载)
    torch.save(model.state_dict(), "bc_distill_policy.pt")






