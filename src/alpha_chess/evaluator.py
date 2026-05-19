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
MATE_SCORE_CP = 100_000


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
        material_value_search_plies: int = 0,
    ) -> None:
        if not 0.0 <= material_value_weight <= 1.0:
            raise ValueError("material_value_weight must be between 0 and 1")
        self.device = resolve_torch_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.material_value_weight = float(material_value_weight)
        self.material_value_search_plies = max(0, int(material_value_search_plies))

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
            value_out += self.material_value_weight * material_value(
                board,
                search_plies=self.material_value_search_plies,
            )
        else:
            value_out = neural_value
        return policy, float(value_out)


def material_value(board: chess.Board, search_plies: int = 0) -> float:
    if search_plies > 0:
        score = _material_quiescence_score(
            board,
            max(0, int(search_plies)),
            board.turn,
            {},
        )
        return float(np.tanh(score / 1200.0))
    return float(np.tanh(_material_score_cp(board, board.turn) / 1200.0))


def _material_quiescence_score(
    board: chess.Board,
    plies: int,
    perspective: chess.Color,
    cache: dict[tuple[str, int, bool], int],
) -> int:
    key = (board.fen(), plies, perspective)
    if key in cache:
        return cache[key]

    stand_pat = _material_score_cp(board, perspective)
    if plies <= 0 or board.is_game_over(claim_draw=True):
        cache[key] = stand_pat
        return stand_pat

    moves = _material_tactical_moves(board)
    if not moves:
        cache[key] = stand_pat
        return stand_pat

    if board.turn == perspective:
        best = stand_pat
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = max(best, _material_quiescence_score(child, plies - 1, perspective, cache))
    else:
        best = stand_pat
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = min(best, _material_quiescence_score(child, plies - 1, perspective, cache))

    cache[key] = best
    return best


def _material_tactical_moves(board: chess.Board) -> list[chess.Move]:
    moves = [
        move
        for move in board.legal_moves
        if board.is_capture(move) or move.promotion or board.gives_check(move)
    ]
    moves.sort(key=lambda move: _move_material_priority(board, move), reverse=True)
    return moves


def _move_material_priority(board: chess.Board, move: chess.Move) -> int:
    captured_value = _captured_piece_value(board, move)
    promotion_value = PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0
    check_bonus = 10_000 if board.gives_check(move) else 0
    moving_piece = board.piece_at(move.from_square)
    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
    return check_bonus + captured_value + promotion_value - moving_value


def _captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return PIECE_VALUES.get(captured.piece_type, 0) if captured else 0


def _material_score_cp(board: chess.Board, perspective: chess.Color) -> int:
    if board.is_checkmate():
        return -MATE_SCORE_CP if board.turn == perspective else MATE_SCORE_CP
    if board.is_game_over(claim_draw=True):
        return 0

    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == perspective else -value
    return score


def load_evaluator(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    material_value_weight: float = 0.0,
    material_value_search_plies: int = 0,
) -> NeuralEvaluator:
    resolved = resolve_torch_device(device)
    return NeuralEvaluator(
        load_checkpoint(checkpoint, map_location=resolved),
        device=resolved,
        material_value_weight=material_value_weight,
        material_value_search_plies=material_value_search_plies,
    )
