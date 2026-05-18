import chess
import pytest

from alpha_chess.evaluator import material_value


def test_material_value_is_from_side_to_move_perspective() -> None:
    white_to_move = chess.Board("7k/8/8/8/8/8/8/Q3K3 w - - 0 1")
    black_to_move = chess.Board("7k/8/8/8/8/8/8/Q3K3 b - - 0 1")

    assert material_value(white_to_move) > 0
    assert material_value(black_to_move) < 0
    assert material_value(white_to_move) == pytest.approx(-material_value(black_to_move))
