"""Policy/value network for AlphaChess."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_chess.chess_env import ACTION_SIZE, NUM_INPUT_PLANES


@dataclass
class ChessNetConfig:
    input_planes: int = NUM_INPUT_PLANES
    channels: int = 128
    blocks: int = 6
    policy_channels: int = 32
    value_channels: int = 16
    dropout: float = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """Small residual policy/value network.

    The policy head emits logits over the full 64 x 73 action space. Illegal
    action masking is handled by the evaluator and MCTS.
    """

    def __init__(self, config: ChessNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChessNetConfig()
        c = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_planes, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(c, self.config.dropout) for _ in range(self.config.blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(c, self.config.policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.policy_channels * 8 * 8, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, self.config.value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.config.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.value_channels * 8 * 8, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
            nn.Tanh(),
        )

    def forward(self, board_bchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(board_bchw)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value

    def compute_loss(
        self,
        board_bchw: torch.Tensor,
        target_policy_ba: torch.Tensor,
        target_value_b: torch.Tensor,
        value_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_logits, value = self.forward(board_bchw)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policy_ba * log_probs).sum(dim=-1).mean()
        target_action = target_policy_ba.argmax(dim=-1)
        policy_acc = (policy_logits.argmax(dim=-1) == target_action).float().mean()
        value_loss = F.mse_loss(value, target_value_b)
        loss = policy_loss + value_weight * value_loss
        return loss, {
            "policy_loss": policy_loss.detach(),
            "policy_acc": policy_acc.detach(),
            "value_loss": value_loss.detach(),
        }

    def compute_loss_from_actions(
        self,
        board_bchw: torch.Tensor,
        target_action_b: torch.Tensor,
        target_value_b: torch.Tensor,
        value_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_logits, value = self.forward(board_bchw)
        policy_loss = F.cross_entropy(policy_logits, target_action_b.long())
        policy_acc = (policy_logits.argmax(dim=-1) == target_action_b.long()).float().mean()
        value_loss = F.mse_loss(value, target_value_b)
        loss = policy_loss + value_weight * value_loss
        return loss, {
            "policy_loss": policy_loss.detach(),
            "policy_acc": policy_acc.detach(),
            "value_loss": value_loss.detach(),
        }


def save_checkpoint(
    path: str | Path,
    model: ChessNet,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ChessNet:
    payload = torch.load(path, map_location=map_location)
    config = ChessNetConfig(**payload.get("config", {}))
    model = ChessNet(config)
    model.load_state_dict(payload["model_state"])
    return model
