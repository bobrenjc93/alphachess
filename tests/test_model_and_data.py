import numpy as np
import pytest
import torch

from alpha_chess.chess_env import ACTION_SIZE, NUM_INPUT_PLANES
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.model import ChessNet, ChessNetConfig


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
    assert float(sample["value"]) == -1.0

    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["policy"].shape == (2, ACTION_SIZE)


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

    with pytest.raises(ValueError, match="data_weights"):
        dataset.source_sample_weights([1.0])


def _write_sparse_npz(path, positions: int) -> None:
    boards = np.zeros((positions, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    actions = np.zeros((positions,), dtype=np.int64)
    values = np.zeros((positions,), dtype=np.float32)
    np.savez_compressed(path, boards=boards, actions=actions, values=values)
