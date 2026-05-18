import numpy as np
import pytest

from alpha_chess.iteration import (
    IterationConfig,
    _build_training_inputs,
    _filter_nonempty_data_dirs,
)


def test_iteration_training_inputs_mix_replay_data() -> None:
    config = IterationConfig(
        replay_data=["teacher", "puzzles"],
        self_play_weight=0.4,
        replay_weights=[0.5, 0.1],
    )

    train_data, data_weights = _build_training_inputs(["selfplay-1", "selfplay-2"], config)

    assert train_data == ["selfplay-1", "selfplay-2", "teacher", "puzzles"]
    assert data_weights == pytest.approx([0.2, 0.2, 0.5, 0.1])


def test_iteration_training_inputs_default_replay_weights() -> None:
    config = IterationConfig(replay_data=["teacher"])

    train_data, data_weights = _build_training_inputs(["selfplay-1", "selfplay-2"], config)

    assert train_data == ["selfplay-1", "selfplay-2", "teacher"]
    assert data_weights == pytest.approx([0.5, 0.5, 1.0])


def test_iteration_training_inputs_validate_replay_weights() -> None:
    config = IterationConfig(replay_data=["teacher", "puzzles"], replay_weights=[1.0])

    with pytest.raises(ValueError, match="replay_weights"):
        _build_training_inputs(["selfplay"], config)


def test_iteration_training_inputs_allow_replay_only() -> None:
    config = IterationConfig(replay_data=["teacher"], replay_weights=[1.0])

    train_data, data_weights = _build_training_inputs([], config)

    assert train_data == ["teacher"]
    assert data_weights == pytest.approx([1.0])


def test_iteration_filters_empty_selfplay_dirs(tmp_path) -> None:
    empty_dir = tmp_path / "empty"
    data_dir = tmp_path / "with-data"
    empty_dir.mkdir()
    data_dir.mkdir()
    np.savez_compressed(data_dir / "game.npz", boards=np.zeros((0, 1)), values=np.zeros((0,)))

    filtered = _filter_nonempty_data_dirs(
        [str(empty_dir), str(data_dir), str(tmp_path / "missing")]
    )

    assert filtered == [str(data_dir)]
