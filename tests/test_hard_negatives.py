import chess
import numpy as np
import torch

from alpha_chess.chess_env import encode_board, move_to_action
from alpha_chess.hard_negatives import (
    HardNegativeConfig,
    _top_wrong_predictions,
    mine_hard_negatives,
)
from alpha_chess.model import ChessNet, ChessNetConfig, save_checkpoint


def test_top_wrong_predictions_marks_only_wrong_top_action() -> None:
    logits = torch.full((2, 6), -10.0)
    logits[0, 3] = 5.0
    logits[1, 4] = 5.0
    targets = torch.tensor([1, 4])

    bad_actions = _top_wrong_predictions(logits, targets)

    assert bad_actions.tolist() == [3, -1]


def test_mine_hard_negatives_writes_model_top_wrong_move(tmp_path) -> None:
    board = chess.Board()
    target = move_to_action(chess.Move.from_uci("e2e4"), board)
    bad = move_to_action(chess.Move.from_uci("d2d4"), board)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "teacher.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        actions=np.asarray([target], dtype=np.int64),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    for param in model.parameters():
        param.data.zero_()
    model.policy_head[-1].bias.data.fill_(-1.0)
    model.policy_head[-1].bias.data[bad] = 10.0
    model.policy_head[-1].bias.data[target] = 1.0
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model)

    paths = mine_hard_negatives(
        HardNegativeConfig(
            checkpoint=str(checkpoint),
            data=str(data_dir),
            out=str(out_dir),
            batch_size=1,
            chunk_size=1,
            device="cpu",
        )
    )

    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=True) as mined:
        assert mined["boards"].shape == (1, 19, 8, 8)
        assert int(mined["actions"][0]) == target
        assert int(mined["bad_actions"][0]) == bad
        assert mined["fens"][0] == board.fen()

    summary = (out_dir / "hard_negative_summary.txt").read_text()
    assert "positions=1" in summary
    assert "hard_negative_positions=1" in summary
    assert "top1_error_rate=1.000000" in summary
