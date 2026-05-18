import chess
import chess.engine
import chess.pgn

from alpha_chess.stockfish_teacher import (
    _matches_player_to_move,
    _resolve_pgn_paths,
    _score_to_value,
    _value_drop_after_move,
)


def test_score_to_value_from_side_to_move() -> None:
    score = chess.engine.PovScore(chess.engine.Cp(300), chess.WHITE)
    assert _score_to_value(score, chess.WHITE) > 0
    assert _score_to_value(score, chess.BLACK) < 0


def test_value_drop_after_move_maps_back_to_original_side() -> None:
    best_value = 0.5
    after_score = chess.engine.PovScore(chess.engine.Cp(120), chess.BLACK)
    assert _value_drop_after_move(best_value, after_score, chess.BLACK) > 0.6


def test_matches_player_to_move_uses_pgn_headers() -> None:
    game = chess.pgn.Game()
    game.headers["White"] = "AlphaChess"
    game.headers["Black"] = "Stockfish"
    board = game.board()

    assert _matches_player_to_move(game, board, "alphachess")

    board.push(chess.Move.from_uci("e2e4"))
    assert not _matches_player_to_move(game, board, "AlphaChess")
    assert _matches_player_to_move(game, board, "Stockfish")


def test_resolve_pgn_paths_accepts_one_or_many() -> None:
    assert [str(path) for path in _resolve_pgn_paths("one.pgn")] == ["one.pgn"]
    assert [str(path) for path in _resolve_pgn_paths(["one.pgn", "two.pgn"])] == [
        "one.pgn",
        "two.pgn",
    ]
