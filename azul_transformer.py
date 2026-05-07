import torch
import torch.nn as nn


class AzulTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, action_dim=180):
        super().__init__()

        # 针对不同区域设置 Embedding 层
        self.factory_emb = nn.Linear(24, d_model)
        self.center_emb = nn.Linear(6, d_model)
        self.pattern_emb = nn.Linear(30, d_model)
        self.wall_emb = nn.Linear(25, d_model)
        self.floor_emb = nn.Linear(42, d_model)
        self.score_emb = nn.Linear(1, d_model)
        self.global_emb = nn.Linear(5, d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dropout=0.1  # 建议加一点 dropout 防止过拟合
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # --- 1. 拆分特征 (根据你的索引) ---
        # 切片逻辑保持不变
        factories = x[:, 0:120].view(-1, 5, 24)
        center = x[:, 120:126].unsqueeze(1)
        my_wall = x[:, 126:151].unsqueeze(1)
        my_patterns = x[:, 151:301].view(-1, 5, 30)
        my_floor = x[:, 301:343].unsqueeze(1)
        my_score = x[:, 343:344].unsqueeze(1)

        opp_wall = x[:, 344:369].unsqueeze(1)
        opp_patterns = x[:, 369:519].view(-1, 5, 30)
        opp_floor = x[:, 519:561].unsqueeze(1)
        opp_score = x[:, 561:562].unsqueeze(1)

        global_feat = x[:, -5:]

        # --- 2. 映射到统一维度并对齐到 [B, N, d_model] ---
        # 注意：对于单 token 区域，必须 unsqueeze(1)
        tokens = [
            self.global_emb(global_feat).unsqueeze(1),  # [B, 1, d_model]
            self.factory_emb(factories),  # [B, 5, d_model]
            self.center_emb(center),  # [B, 1, d_model]
            self.wall_emb(my_wall),  # [B, 1, d_model]
            self.pattern_emb(my_patterns),  # [B, 5, d_model]
            self.floor_emb(my_floor),  # [B, 1, d_model]
            self.score_emb(my_score),  # [B, 1, d_model]
            self.wall_emb(opp_wall),  # [B, 1, d_model] 对手复用同一个 emb
            self.pattern_emb(opp_patterns),  # [B, 5, d_model] 对手复用
            self.floor_emb(opp_floor),  # [B, 1, d_model] 对手复用
            self.score_emb(opp_score),  # [B, 1, d_model] 对手复用
        ]

        # 拼接所有 tokens -> [B, 23, d_model] (1+5+1+1+5+1+1+1+5+1+1 = 23个Token)
        combined_tokens = torch.cat(tokens, dim=1)

        # --- 3. Transformer 处理 ---
        feat_seq = self.transformer(combined_tokens)

        # 4. 聚合信息
        # 既然你有 Global Token (索引0)，也可以尝试 feat = feat_seq[:, 0, :]
        # 但 mean 均值池化通常更稳健
        feat = feat_seq.mean(dim=1)

        policy_logits = self.policy_head(feat)
        value_logit = self.value_head(feat).squeeze(-1)

        return policy_logits, value_logit