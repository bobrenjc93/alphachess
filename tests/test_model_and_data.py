import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alpha_chess.chess_env import ACTION_SIZE, NUM_INPUT_PLANES, encode_board, move_to_action
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.model import ChessNet, ChessNetConfig, save_checkpoint
from alpha_chess.train import (
    ValidateConfig,
    _compute_batch_loss,
    _evaluate_loss,
    _legal_action_mask_from_fens,
    validate,
)


def test_model_forward_shapes() -> None:
    model = ChessNet(ChessNetConfig(channels=16, blocks=1))
    boards = torch.zeros(2, NUM_INPUT_PLANES, 8, 8)
    policy, value = model(boards)
    assert policy.shape == (2, ACTION_SIZE)
    assert value.shape == (2,)

    actions = torch.zeros(2, dtype=torch.long)
    values = torch.zeros(2)
    loss, parts = model.compute_loss_from_actions(boards, actions, values)
    assert loss.ndim == 0
    assert "policy_acc" in parts


def test_self_play_dataset_loads_npz(tmp_path) -> None:
    boards = np.zeros((3, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    policies = np.zeros((3, ACTION_SIZE), dtype=np.float32)
    policies[:, 0] = 1.0
    values = np.asarray([1.0, -1.0, 0.0], dtype=np.float32)
    np.savez_compressed(tmp_path / "game.npz", boards=boards, policies=policies, values=values)

    dataset = SelfPlayDataset(tmp_path, in_memory=True)
    sample = dataset[1]
    assert len(dataset) == 3
    assert sample["board"].shape == (NUM_INPUT_PLANES, 8, 8)
    assert sample["policy"].shape == (ACTION_SIZE,)
    assert int(sample["source_id"]) == 0
    assert float(sample["value"]) == -1.0

    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["policy"].shape == (2, ACTION_SIZE)
    assert batch["source_id"].shape == (2,)


def test_source_sample_weights_balance_input_paths(tmp_path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_sparse_npz(source_a / "a.npz", positions=2)
    _write_sparse_npz(source_b / "b.npz", positions=6)

    dataset = SelfPlayDataset([source_a, source_b])
    weights = dataset.source_sample_weights([0.75, 0.25])

    assert len(dataset) == 8
    assert weights.dtype == torch.double
    assert torch.isclose(weights[:2].sum(), torch.tensor(0.75, dtype=torch.double))
    assert torch.isclose(weights[2:].sum(), torch.tensor(0.25, dtype=torch.double))
    assert int(dataset[0]["source_id"]) == 0
    assert int(dataset[2]["source_id"]) == 1

    with pytest.raises(ValueError, match="data_weights"):
        dataset.source_sample_weights([1.0])


def test_dataset_collates_fens_for_legal_policy_loss(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    np.savez_compressed(
        tmp_path / "fen.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        actions=np.asarray([move_to_action(move, board)], dtype=np.int64),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    dataset = SelfPlayDataset(tmp_path, in_memory=True)
    sample = dataset[0]
    batch = collate_samples([sample])

    assert sample["fen"] == board.fen()
    assert batch["fen"] == [board.fen()]


def test_legal_policy_loss_masks_to_legal_actions() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    action = move_to_action(move, board)
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    batch = {
        "board": torch.from_numpy(np.asarray([encode_board(board)], dtype=np.float32)),
        "action": torch.tensor([action], dtype=torch.long),
        "value": torch.zeros(1),
        "fen": [board.fen()],
    }

    mask = _legal_action_mask_from_fens([board.fen()], torch.device("cpu"))
    loss, parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=1.0,
        legal_policy_loss=True,
    )

    assert int(mask.sum()) == 20
    assert bool(mask[0, action])
    assert loss.ndim == 0
    assert "policy_acc" in parts


def test_evaluate_loss_reports_source_metrics(tmp_path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_sparse_npz(source_a / "a.npz", positions=3)
    _write_sparse_npz(source_b / "b.npz", positions=5)
    dataset = SelfPlayDataset([source_a, source_b])
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_samples)
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))

    metrics = _evaluate_loss(
        model,
        loader,
        torch.device("cpu"),
        value_weight=1.0,
        source_names=dataset.source_names,
    )

    assert metrics["val_examples"] == 8.0
    assert metrics["val_source_0_examples"] == 3.0
    assert metrics["val_source_1_examples"] == 5.0
    assert "val_source_0_policy_acc" in metrics
    assert "val_source_1_policy_acc" in metrics


def test_validate_checkpoint_reports_metrics(tmp_path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_sparse_npz(source_a / "a.npz", positions=3)
    _write_sparse_npz(source_b / "b.npz", positions=5)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, ChessNet(ChessNetConfig(channels=8, blocks=1)))

    metrics = validate(
        ValidateConfig(
            checkpoint=str(checkpoint),
            data=[str(source_a), str(source_b)],
            batch_size=4,
            device="cpu",
        )
    )

    assert metrics["val_examples"] == 8.0
    assert metrics["val_source_0_examples"] == 3.0
    assert metrics["val_source_1_examples"] == 5.0


def _write_sparse_npz(path, positions: int) -> None:
    boards = np.zeros((positions, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    actions = np.zeros((positions,), dtype=np.int64)
    values = np.zeros((positions,), dtype=np.float32)
    np.savez_compressed(path, boards=boards, actions=actions, values=values)
