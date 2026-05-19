import chess
import torch

from alpha_chess.evaluate import EvalConfig, EvalGameRecord, evaluate_checkpoint, write_eval_pgns
from alpha_chess.model import ChessNet, ChessNetConfig, save_checkpoint


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


def test_parallel_evaluation_matches_serial(tmp_path) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, ChessNet(ChessNetConfig(channels=8, blocks=1)))
    base_config = EvalConfig(
        checkpoint=str(checkpoint),
        games=2,
        simulations=0,
        opponent="uniform",
        device="cpu",
        seed=7,
        max_plies=2,
    )

    serial = evaluate_checkpoint(base_config)
    parallel = evaluate_checkpoint(EvalConfig(**{**base_config.__dict__, "workers": 2}))

    assert parallel == serial
