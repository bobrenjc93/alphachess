"""Chess state encoding and AlphaZero-style action mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chess
import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE

QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
PROMOTION_DELTAS: tuple[tuple[int, int], ...] = ((-1, 1), (0, 1), (1, 1))
UNDERPROMOTIONS: tuple[chess.PieceType, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

QUEEN_PLANES = len(QUEEN_DIRECTIONS) * 7
KNIGHT_PLANES = len(KNIGHT_DELTAS)
PROMOTION_PLANES = len(PROMOTION_DELTAS) * len(UNDERPROMOTIONS)
ACTION_PLANES = QUEEN_PLANES + KNIGHT_PLANES + PROMOTION_PLANES
ACTION_SIZE = NUM_SQUARES * ACTION_PLANES

PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
PIECE_TO_PLANE = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
NUM_INPUT_PLANES = 19


@dataclass(frozen=True)
class DecodedAction:
    """Human-readable action decode for logging and tests."""

    from_square: chess.Square
    to_square: chess.Square
    promotion: chess.PieceType | None

    def to_move(self) -> chess.Move:
        return chess.Move(self.from_square, self.to_square, promotion=self.promotion)


def rotate_square(square: chess.Square) -> chess.Square:
    """Rotate a square 180 degrees."""

    return chess.square(7 - chess.square_file(square), 7 - chess.square_rank(square))


def orient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map an actual square into the side-to-move frame."""

    return square if turn == chess.WHITE else rotate_square(square)


def unorient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a side-to-move-frame square back to the actual board."""

    return square if turn == chess.WHITE else rotate_square(square)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _square_from_file_rank(file_idx: int, rank_idx: int) -> chess.Square | None:
    if 0 <= file_idx < BOARD_SIZE and 0 <= rank_idx < BOARD_SIZE:
        return chess.square(file_idx, rank_idx)
    return None


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """Encode a legal move as an integer action.

    The origin square and move direction are expressed from the side-to-move
    frame, so black positions are rotated 180 degrees before encoding.
    """

    from_sq = orient_square(move.from_square, board.turn)
    to_sq = orient_square(move.to_square, board.turn)
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)
    df = to_file - from_file
    dr = to_rank - from_rank

    if move.promotion in UNDERPROMOTIONS:
        try:
            direction_index = PROMOTION_DELTAS.index((df, dr))
            piece_index = UNDERPROMOTIONS.index(move.promotion)
        except ValueError as exc:
            raise ValueError(f"Unsupported underpromotion move: {move.uci()}") from exc
        plane = QUEEN_PLANES + KNIGHT_PLANES + direction_index * len(UNDERPROMOTIONS) + piece_index
    elif (df, dr) in KNIGHT_DELTAS:
        plane = QUEEN_PLANES + KNIGHT_DELTAS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        if distance < 1 or distance > 7:
            raise ValueError(f"Move has invalid distance: {move.uci()}")
        if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
            raise ValueError(f"Move is not queen-like, knight, or promotion: {move.uci()}")
        direction = (_sign(df), _sign(dr))
        try:
            direction_index = QUEEN_DIRECTIONS.index(direction)
        except ValueError as exc:
            raise ValueError(f"Move has unsupported direction: {move.uci()}") from exc
        plane = direction_index * 7 + (distance - 1)

    return from_sq * ACTION_PLANES + plane


def decode_action(action: int, board: chess.Board) -> DecodedAction | None:
    """Decode an action into a move candidate without checking legality."""

    if action < 0 or action >= ACTION_SIZE:
        return None

    from_sq_oriented = action // ACTION_PLANES
    plane = action % ACTION_PLANES
    from_file = chess.square_file(from_sq_oriented)
    from_rank = chess.square_rank(from_sq_oriented)
    promotion: chess.PieceType | None = None

    if plane < QUEEN_PLANES:
        direction = QUEEN_DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        df = direction[0] * distance
        dr = direction[1] * distance
    elif plane < QUEEN_PLANES + KNIGHT_PLANES:
        df, dr = KNIGHT_DELTAS[plane - QUEEN_PLANES]
    else:
        under_plane = plane - QUEEN_PLANES - KNIGHT_PLANES
        direction_index = under_plane // len(UNDERPROMOTIONS)
        piece_index = under_plane % len(UNDERPROMOTIONS)
        df, dr = PROMOTION_DELTAS[direction_index]
        promotion = UNDERPROMOTIONS[piece_index]

    to_sq_oriented = _square_from_file_rank(from_file + df, from_rank + dr)
    if to_sq_oriented is None:
        return None

    from_sq = unorient_square(from_sq_oriented, board.turn)
    to_sq = unorient_square(to_sq_oriented, board.turn)

    piece = board.piece_at(from_sq)
    if promotion is None and piece is not None and piece.piece_type == chess.PAWN:
        if chess.square_rank(to_sq_oriented) == 7:
            promotion = chess.QUEEN

    return DecodedAction(from_sq, to_sq, promotion)


def action_to_move(action: int, board: chess.Board) -> chess.Move | None:
    """Decode an action and return it only if legal in ``board``."""

    decoded = decode_action(action, board)
    if decoded is None:
        return None
    move = decoded.to_move()
    return move if move in board.legal_moves else None


def legal_actions(board: chess.Board) -> list[int]:
    """Return all legal actions for a board."""

    return [move_to_action(move, board) for move in board.legal_moves]


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of legal actions."""

    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for action in legal_actions(board):
        mask[action] = True
    return mask


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as planes from the side-to-move perspective.

    Planes 0-5 are current-player pieces, 6-11 are opponent pieces, followed by:
    side-to-move color, current/opponent castling rights, halfmove clock, and
    fullmove number.
    """

    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for square, piece in board.piece_map().items():
        oriented = orient_square(square, board.turn)
        file_idx = chess.square_file(oriented)
        rank_idx = chess.square_rank(oriented)
        piece_offset = PIECE_TO_PLANE[piece.piece_type]
        color_offset = 0 if piece.color == board.turn else 6
        planes[color_offset + piece_offset, rank_idx, file_idx] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = float(board.has_kingside_castling_rights(board.turn))
    planes[14, :, :] = float(board.has_queenside_castling_rights(board.turn))
    planes[15, :, :] = float(board.has_kingside_castling_rights(not board.turn))
    planes[16, :, :] = float(board.has_queenside_castling_rights(not board.turn))
    planes[17, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[18, :, :] = min(board.fullmove_number, 200) / 200.0
    return planes


def terminal_value(board: chess.Board) -> float | None:
    """Return terminal value from side-to-move perspective, or ``None``."""

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def result_value_for_color(board: chess.Board, color: chess.Color) -> float:
    """Return final game value from ``color`` perspective."""

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


def actions_to_policy(actions: Iterable[int], probabilities: Iterable[float]) -> np.ndarray:
    """Create a dense policy vector from sparse action probabilities."""

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    for action, probability in zip(actions, probabilities):
        policy[action] = float(probability)
    total = float(policy.sum())
    if total > 0:
        policy /= total
    return policy
