"""Policy/value evaluators used by MCTS."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import chess
import numpy as np
import torch

from alpha_chess.chess_env import ACTION_SIZE, encode_board, legal_actions
from alpha_chess.model import ChessNet, load_checkpoint

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


class Evaluator(Protocol):
    def __call__(self, board: chess.Board) -> tuple[np.ndarray, float]: ...


class UniformEvaluator:
    """Uniform legal policy with neutral value."""

    def __call__(self, board: chess.Board) -> tuple[np.ndarray, float]:
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        actions = legal_actions(board)
        if actions:
            policy[actions] = 1.0 / len(actions)
        return policy, 0.0


def resolve_torch_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class NeuralEvaluator:
    """Torch policy/value evaluator with legal action masking."""

    def __init__(
        self,
        model: ChessNet,
        device: str | torch.device = "cpu",
        material_value_weight: float = 0.0,
    ) -> None:
        if not 0.0 <= material_value_weight <= 1.0:
            raise ValueError("material_value_weight must be between 0 and 1")
        self.device = resolve_torch_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.material_value_weight = float(material_value_weight)

    @torch.no_grad()
    def __call__(self, board: chess.Board) -> tuple[np.ndarray, float]:
        features = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(features)
        logits_np = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
        actions = legal_actions(board)
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        if actions:
            legal_logits = logits_np[actions]
            legal_logits -= np.max(legal_logits)
            probs = np.exp(legal_logits)
            total = float(probs.sum())
            if total <= 0 or not np.isfinite(total):
                policy[actions] = 1.0 / len(actions)
            else:
                policy[actions] = (probs / total).astype(np.float32)
        neural_value = float(value.item())
        if self.material_value_weight:
            value_out = (1.0 - self.material_value_weight) * neural_value
            value_out += self.material_value_weight * material_value(board)
        else:
            value_out = neural_value
        return policy, float(value_out)


def material_value(board: chess.Board) -> float:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == board.turn else -value
    return float(np.tanh(score / 1200.0))


def load_evaluator(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    material_value_weight: float = 0.0,
) -> NeuralEvaluator:
    resolved = resolve_torch_device(device)
    return NeuralEvaluator(
        load_checkpoint(checkpoint, map_location=resolved),
        device=resolved,
        material_value_weight=material_value_weight,
    )
