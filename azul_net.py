import torch
import torch.nn as nn


class AzulNet(nn.Module):
    def __init__(self, obs_dim=567, action_dim=180):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.trunk(x)
        policy_logits = self.policy_head(feat)          # [B, 180]
        value_logit = self.value_head(feat).squeeze(-1) # [B]
        return policy_logits, value_logit