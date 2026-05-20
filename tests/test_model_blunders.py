import chess
import chess.engine
import numpy as np
import torch

from alpha_chess.chess_env import ACTION_SIZE, encode_board, move_to_action
from alpha_chess.model_blunders import (
    ModelBlunderConfig,
    _stockfish_bad_actions,
    mine_model_blunders,
)


class _FakeEngine:
    def __init__(self, bad_move: chess.Move) -> None:
        self.bad_move = bad_move

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        del limit
        last_move = board.peek()
        cp = 600 if last_move == self.bad_move else -600
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}


def test_stockfish_bad_actions_keeps_only_value_dropping_model_moves() -> None:
    board = chess.Board()
    target_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    bad_move = chess.Move.from_uci("d2d4")
    bad_action = move_to_action(bad_move, board)
    safe_action = move_to_action(chess.Move.from_uci("g1f3"), board)

    bad_actions, value_deltas = _stockfish_bad_actions(
        engine=_FakeEngine(bad_move),
        board=board,
        ranked_actions=[bad_action, safe_action, target_action],
        target_action=target_action,
        best_value=0.0,
        limit=chess.engine.Limit(time=0.01),
        min_value_delta=0.1,
        max_bad_actions=2,
    )

    assert bad_actions == [bad_action]
    assert value_deltas[0] > 0.1


def test_mine_model_blunders_writes_stockfish_confirmed_bad_action(
    monkeypatch,
    tmp_path,
) -> None:
    board = chess.Board()
    target_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    bad_move = chess.Move.from_uci("d2d4")
    bad_action = move_to_action(bad_move, board)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "teacher.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        actions=np.asarray([target_action], dtype=np.int64),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    class FakeModel(torch.nn.Module):
        def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.full((boards.shape[0], ACTION_SIZE), -10.0)
            logits[:, bad_action] = 10.0
            logits[:, target_action] = 1.0
            return logits, torch.zeros(boards.shape[0])

    class FakeEngineContext:
        def __enter__(self) -> _FakeEngine:
            return _FakeEngine(bad_move)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr("alpha_chess.model_blunders.load_checkpoint", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        "alpha_chess.model_blunders.chess.engine.SimpleEngine.popen_uci",
        lambda _path: FakeEngineContext(),
    )

    paths = mine_model_blunders(
        ModelBlunderConfig(
            checkpoint="checkpoint.pt",
            data=str(data_dir),
            out=str(out_dir),
            max_positions=1,
            min_value_delta=0.1,
            device="cpu",
        )
    )

    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=True) as mined:
        assert mined["boards"].shape == (1, 19, 8, 8)
        assert int(mined["actions"][0]) == target_action
        assert int(mined["bad_actions"][0, 0]) == bad_action
        assert float(mined["value_deltas"][0, 0]) > 0.1

    summary = (out_dir / "model_blunder_summary.txt").read_text()
    assert "positions_seen=1" in summary
    assert "model_wrong_positions=1" in summary
    assert "blunder_positions=1" in summary
    assert "bad_actions=1" in summary
