import torch
import torch.nn as nn


class AzulTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, action_dim=180):
        super().__init__()
        self.num_tokens = 23

        self.factory_emb = nn.Linear(24, d_model)
        self.center_emb = nn.Linear(6, d_model)
        self.pattern_emb = nn.Linear(30, d_model)
        self.wall_emb = nn.Linear(25, d_model)
        self.floor_emb = nn.Linear(42, d_model)
        self.score_emb = nn.Linear(1, d_model)
        self.global_emb = nn.Linear(5, d_model)
        self.position_emb = nn.Embedding(self.num_tokens, d_model)
        self.token_type_emb = nn.Embedding(11, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        token_type_ids = [0]
        token_type_ids.extend([1] * 5)
        token_type_ids.extend([2])
        token_type_ids.extend([3])
        token_type_ids.extend([4] * 5)
        token_type_ids.extend([5])
        token_type_ids.extend([6])
        token_type_ids.extend([7])
        token_type_ids.extend([8] * 5)
        token_type_ids.extend([9])
        token_type_ids.extend([10])
        self.register_buffer(
            "token_type_ids",
            torch.tensor(token_type_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
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

        tokens = [
            self.global_emb(global_feat).unsqueeze(1),
            self.factory_emb(factories),
            self.center_emb(center),
            self.wall_emb(my_wall),
            self.pattern_emb(my_patterns),
            self.floor_emb(my_floor),
            self.score_emb(my_score),
            self.wall_emb(opp_wall),
            self.pattern_emb(opp_patterns),
            self.floor_emb(opp_floor),
            self.score_emb(opp_score),
        ]

        combined_tokens = torch.cat(tokens, dim=1)
        batch_size = combined_tokens.size(0)

        position_ids = torch.arange(self.num_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(batch_size, -1)
        combined_tokens = (
            combined_tokens
            + self.position_emb(position_ids)
            + self.token_type_emb(token_type_ids)
        )

        feat_seq = self.transformer(combined_tokens)
        feat = feat_seq[:, 0, :]

        policy_logits = self.policy_head(feat)
        value_logit = self.value_head(feat).squeeze(-1)

        return policy_logits, value_logit
