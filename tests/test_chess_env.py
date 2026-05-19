import chess
import numpy as np

from alpha_chess.chess_env import (
    ACTION_SIZE,
    NUM_INPUT_PLANES,
    action_to_move,
    color_mirror_action,
    color_mirror_board,
    color_mirror_move,
    color_mirror_policy,
    encode_board,
    legal_action_mask,
    move_to_action,
)


def assert_roundtrip(board: chess.Board) -> None:
    for move in board.legal_moves:
        action = move_to_action(move, board)
        assert 0 <= action < ACTION_SIZE
        assert action_to_move(action, board) == move


def test_starting_position_roundtrips_all_legal_moves() -> None:
    assert_roundtrip(chess.Board())


def test_castling_roundtrip() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert chess.Move.from_uci("e1g1") in board.legal_moves
    assert chess.Move.from_uci("e1c1") in board.legal_moves
    assert_roundtrip(board)


def test_promotions_roundtrip_for_both_colors() -> None:
    white = chess.Board("4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1")
    black = chess.Board("4k3/P6P/8/8/8/8/p6p/4K3 b - - 0 1")
    assert_roundtrip(white)
    assert_roundtrip(black)


def test_encode_board_shape_and_mask() -> None:
    board = chess.Board()
    encoded = encode_board(board)
    mask = legal_action_mask(board)
    assert encoded.shape == (NUM_INPUT_PLANES, 8, 8)
    assert encoded.dtype == np.float32
    assert mask.shape == (ACTION_SIZE,)
    assert int(mask.sum()) == board.legal_moves.count()


def test_color_mirror_maps_moves_and_policies() -> None:
    board = chess.Board()
    mirrored = color_mirror_board(board)
    move = chess.Move.from_uci("g1f3")
    mirrored_move = color_mirror_move(move)
    action = move_to_action(move, board)
    mirrored_action = color_mirror_action(action, board)

    assert mirrored.turn == chess.BLACK
    assert mirrored_move == chess.Move.from_uci("g8f6")
    assert action_to_move(mirrored_action, mirrored) == mirrored_move

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[action] = 1.0
    mirrored_policy = color_mirror_policy(policy, board)
    assert mirrored_policy[mirrored_action] == 1.0
    assert mirrored_policy.sum() == 1.0
