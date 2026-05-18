import chess

from alpha_chess.evaluate import EvalGameRecord, write_eval_pgns


def test_write_eval_pgns(tmp_path) -> None:
    board = chess.Board()
    board.push_san("e4")
    path = write_eval_pgns(
        tmp_path / "eval.pgn",
        [
            EvalGameRecord(
                board=board,
                model_color=chess.WHITE,
                score=0.5,
                opponent_name="test-opponent",
            )
        ],
    )

    text = path.read_text()
    assert '[White "AlphaChess"]' in text
    assert '[Black "test-opponent"]' in text
    assert '[AlphaChessScore "0.5"]' in text
    assert "1. e4" in text
