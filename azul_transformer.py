import torch
import torch.nn as nn

from config import VALUE_VECTOR_DIM


class AzulTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=512, action_dim=180):
        super().__init__()
        self.num_factory_tokens = 9
        self.num_player_tokens = 4
        self.tokens_per_player = 8
        self.num_tokens = 1 + self.num_factory_tokens + 1 + self.num_player_tokens * self.tokens_per_player

        self.factory_emb = nn.Linear(24, d_model)
        self.center_emb = nn.Linear(6, d_model)
        self.player_meta_emb = nn.Linear(4, d_model)
        self.pattern_emb = nn.Linear(30, d_model)
        self.wall_emb = nn.Linear(25, d_model)
        self.floor_emb = nn.Linear(42, d_model)
        self.global_emb = nn.Linear(2, d_model)
        self.position_emb = nn.Embedding(self.num_tokens, d_model)
        self.token_type_emb = nn.Embedding(5, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, VALUE_VECTOR_DIM),
        )

        token_type_ids = [0]
        token_type_ids.extend([1] * self.num_factory_tokens)
        token_type_ids.extend([2])
        for _ in range(self.num_player_tokens):
            token_type_ids.extend([3, 4, 4, 4, 4, 4, 4, 4])
        self.register_buffer(
            "token_type_ids",
            torch.tensor(token_type_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
        ptr = 0
        factories = x[:, ptr:ptr + 216].view(-1, self.num_factory_tokens, 24)
        ptr += 216
        center = x[:, ptr:ptr + 6].unsqueeze(1)
        ptr += 6

        player_tokens = []
        for _ in range(self.num_player_tokens):
            player_meta = x[:, ptr:ptr + 4].unsqueeze(1)
            ptr += 4
            player_wall = x[:, ptr:ptr + 25].unsqueeze(1)
            ptr += 25
            player_patterns = x[:, ptr:ptr + 150].view(-1, 5, 30)
            ptr += 150
            player_floor = x[:, ptr:ptr + 42].unsqueeze(1)
            ptr += 42

            player_tokens.extend([
                self.player_meta_emb(player_meta),
                self.wall_emb(player_wall),
                self.pattern_emb(player_patterns),
                self.floor_emb(player_floor),
            ])

        global_feat = x[:, ptr:ptr + 2]

        tokens = [self.global_emb(global_feat).unsqueeze(1), self.factory_emb(factories), self.center_emb(center), *player_tokens]

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
        value_logit = self.value_head(feat)

        return policy_logits, value_logit
