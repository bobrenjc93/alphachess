import chess
import chess.engine

from alpha_chess.stockfish_teacher import _score_to_value, _value_drop_after_move


def test_score_to_value_from_side_to_move() -> None:
    score = chess.engine.PovScore(chess.engine.Cp(300), chess.WHITE)
    assert _score_to_value(score, chess.WHITE) > 0
    assert _score_to_value(score, chess.BLACK) < 0


def test_value_drop_after_move_maps_back_to_original_side() -> None:
    best_value = 0.5
    after_score = chess.engine.PovScore(chess.engine.Cp(120), chess.BLACK)
    assert _value_drop_after_move(best_value, after_score, chess.BLACK) > 0.6
