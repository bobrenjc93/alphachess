import numpy as np
import torch

from alpha_chess.chess_env import ACTION_SIZE, NUM_INPUT_PLANES
from alpha_chess.dataset import SelfPlayDataset
from alpha_chess.model import ChessNet, ChessNetConfig


def test_model_forward_shapes() -> None:
    model = ChessNet(ChessNetConfig(channels=16, blocks=1))
    boards = torch.zeros(2, NUM_INPUT_PLANES, 8, 8)
    policy, value = model(boards)
    assert policy.shape == (2, ACTION_SIZE)
    assert value.shape == (2,)


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
