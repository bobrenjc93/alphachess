import chess
import pytest

from alpha_chess.evaluator import material_value


def test_material_value_is_from_side_to_move_perspective() -> None:
    white_to_move = chess.Board("7k/8/8/8/8/8/8/Q3K3 w - - 0 1")
    black_to_move = chess.Board("7k/8/8/8/8/8/8/Q3K3 b - - 0 1")

    assert material_value(white_to_move) > 0
    assert material_value(black_to_move) < 0
    assert material_value(white_to_move) == pytest.approx(-material_value(black_to_move))


def test_material_value_search_sees_forced_recapture() -> None:
    board = chess.Board("r1b1kbnr/p4ppp/2p1p3/2pp4/2P1PB1P/3P1N2/Pq1N1PP1/R2QK2R b KQkq - 0 9")
    board.push(chess.Move.from_uci("b2a1"))

    assert material_value(board, search_plies=1) > material_value(board)


def test_material_value_search_sees_quiet_mate_check() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2")
    mate = chess.Move.from_uci("d8h4")

    assert board.san(mate).endswith("#")
    assert material_value(board, search_plies=1) > 0.99
