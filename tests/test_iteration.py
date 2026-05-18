import pytest

from alpha_chess.iteration import IterationConfig, _build_training_inputs


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
