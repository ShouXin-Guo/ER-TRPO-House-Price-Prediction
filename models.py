from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def activation_from_name(name: str) -> nn.Module:
    key = name.lower().replace("_", " ").strip()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "linear": nn.Identity,
        "leaky relu": lambda: nn.LeakyReLU(0.01),
        "leakyrelu": lambda: nn.LeakyReLU(0.01),
    }
    if key not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[key]()


class PriceFeatureModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lstm_layers: int,
        mlp_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        blocks: list[nn.Module] = []
        width = hidden_dim
        for _ in range(max(1, mlp_layers)):
            blocks.extend([nn.Linear(width, hidden_dim), activation_from_name(activation), nn.Dropout(dropout)])
            width = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.price_head = nn.Linear(hidden_dim, 1)
        self.relevance_head = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output[:, -1, :]

    def heads_from_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(latent)
        price = self.price_head(h).squeeze(-1)
        relevance = self.relevance_head(h)
        return price, relevance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        price, relevance = self.heads_from_latent(latent)
        return price, relevance, latent

    def predict_from_masked_latent(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = latent * mask
        price, _ = self.heads_from_latent(masked)
        return price


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(state))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
