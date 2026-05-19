import json

import numpy as np
import pytest

from alpha_chess.iteration import (
    IterationConfig,
    _build_training_inputs,
    _filter_nonempty_data_dirs,
    run_iterations,
)


def test_iteration_training_inputs_mix_replay_data() -> None:
    config = IterationConfig(
        replay_data=["teacher", "puzzles"],
        self_play_weight=0.4,
        replay_weights=[0.5, 0.1],
    )

    train_data, data_weights, source_policy_weights = _build_training_inputs(
        ["selfplay-1", "selfplay-2"],
        config,
    )

    assert train_data == ["selfplay-1", "selfplay-2", "teacher", "puzzles"]
    assert data_weights == pytest.approx([0.2, 0.2, 0.5, 0.1])
    assert source_policy_weights is None


def test_iteration_training_inputs_default_replay_weights() -> None:
    config = IterationConfig(replay_data=["teacher"])

    train_data, data_weights, source_policy_weights = _build_training_inputs(
        ["selfplay-1", "selfplay-2"],
        config,
    )

    assert train_data == ["selfplay-1", "selfplay-2", "teacher"]
    assert data_weights == pytest.approx([0.5, 0.5, 1.0])
    assert source_policy_weights is None


def test_iteration_training_inputs_validate_replay_weights() -> None:
    config = IterationConfig(replay_data=["teacher", "puzzles"], replay_weights=[1.0])

    with pytest.raises(ValueError, match="replay_weights"):
        _build_training_inputs(["selfplay"], config)


def test_iteration_training_inputs_allow_replay_only() -> None:
    config = IterationConfig(replay_data=["teacher"], replay_weights=[1.0])

    train_data, data_weights, source_policy_weights = _build_training_inputs([], config)

    assert train_data == ["teacher"]
    assert data_weights == pytest.approx([1.0])
    assert source_policy_weights is None


def test_iteration_training_inputs_build_source_policy_weights() -> None:
    config = IterationConfig(
        replay_data=["teacher", "puzzles"],
        self_play_policy_weight=0.0,
        replay_policy_weights=[1.0, 0.5],
    )

    train_data, data_weights, source_policy_weights = _build_training_inputs(
        ["selfplay-1", "selfplay-2"],
        config,
    )

    assert train_data == ["selfplay-1", "selfplay-2", "teacher", "puzzles"]
    assert data_weights == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert source_policy_weights == pytest.approx([0.0, 0.0, 1.0, 0.5])


def test_iteration_training_inputs_validate_replay_policy_weights() -> None:
    config = IterationConfig(
        replay_data=["teacher", "puzzles"],
        replay_policy_weights=[1.0],
    )

    with pytest.raises(ValueError, match="replay_policy_weights"):
        _build_training_inputs(["selfplay"], config)


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


def test_iteration_uses_checkpoint_self_play_workers(monkeypatch, tmp_path) -> None:
    calls = {}
    train_calls = []

    def fake_generate_from_checkpoint(
        checkpoint,
        out_dir,
        self_play_config,
        *,
        device,
        material_value_weight,
        material_value_search_plies,
    ):
        out_path = out_dir / "game.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, boards=np.zeros((0, 1)), values=np.zeros((0,)))
        calls.update(
            {
                "checkpoint": checkpoint,
                "self_play_config": self_play_config,
                "device": device,
                "material_value_weight": material_value_weight,
                "material_value_search_plies": material_value_search_plies,
            }
        )
        return [out_path]

    def fake_generate_self_play(*_args, **_kwargs):
        raise AssertionError("single-evaluator self-play path should not be used")

    def fake_train(train_config):
        train_calls.append(train_config)
        candidate = tmp_path / "candidate.pt"
        candidate.write_bytes(b"checkpoint")
        return candidate

    monkeypatch.setattr(
        "alpha_chess.iteration.generate_self_play_from_checkpoint",
        fake_generate_from_checkpoint,
    )
    monkeypatch.setattr("alpha_chess.iteration.generate_self_play", fake_generate_self_play)
    monkeypatch.setattr("alpha_chess.iteration.train", fake_train)
    monkeypatch.setattr(
        "alpha_chess.iteration.evaluate_checkpoint_from_dict",
        lambda _config: {"score_rate": 1.0},
    )

    run_dir = tmp_path / "run"
    config = IterationConfig(
        run_dir=str(run_dir),
        iterations=1,
        checkpoint="parent.pt",
        games=2,
        self_play_workers=2,
        simulations=3,
        policy_prior_temperature=2.0,
        material_value_weight=0.15,
        material_value_search_plies=2,
        device="cpu",
        self_play_policy_weight=0.0,
    )

    league_path = run_iterations(config)

    assert league_path == run_dir / "league.json"
    assert calls["checkpoint"] == "parent.pt"
    assert calls["self_play_config"].games == 2
    assert calls["self_play_config"].workers == 2
    assert calls["self_play_config"].simulations == 3
    assert calls["self_play_config"].policy_prior_temperature == pytest.approx(2.0)
    assert calls["device"] == "cpu"
    assert calls["material_value_weight"] == pytest.approx(0.15)
    assert calls["material_value_search_plies"] == 2
    assert train_calls[0].data == [str(run_dir / "selfplay" / "iter_0001")]
    assert train_calls[0].source_policy_weights == [0.0]


def test_iteration_stockfish_gate_can_block_promotion(monkeypatch, tmp_path) -> None:
    eval_calls = []

    def fake_generate_from_checkpoint(
        _checkpoint,
        out_dir,
        _self_play_config,
        *,
        device,
        material_value_weight,
        material_value_search_plies,
    ):
        assert device == "cpu"
        assert material_value_weight == pytest.approx(0.15)
        assert material_value_search_plies == 2
        out_path = out_dir / "game.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, boards=np.zeros((0, 1)), values=np.zeros((0,)))
        return [out_path]

    def fake_train(_train_config):
        candidate = tmp_path / "candidate.pt"
        candidate.write_bytes(b"checkpoint")
        return candidate

    def fake_evaluate(config):
        eval_calls.append(config)
        if len(eval_calls) == 1:
            return {"score_rate": 1.0}
        assert config["opponent"] == "stockfish"
        assert config["engine_path"] == "tools/stockfish/bin/stockfish"
        assert config["games"] == 1
        assert config["simulations"] == 16
        assert config["pgn_out"].endswith("eval/iter_0001_stockfish_gate.pgn")
        return {"score_rate": 0.0}

    monkeypatch.setattr(
        "alpha_chess.iteration.generate_self_play_from_checkpoint",
        fake_generate_from_checkpoint,
    )
    monkeypatch.setattr("alpha_chess.iteration.train", fake_train)
    monkeypatch.setattr("alpha_chess.iteration.evaluate_checkpoint_from_dict", fake_evaluate)

    run_dir = tmp_path / "run"
    league_path = run_iterations(
        IterationConfig(
            run_dir=str(run_dir),
            iterations=1,
            checkpoint="parent.pt",
            games=1,
            self_play_workers=2,
            material_value_weight=0.15,
            material_value_search_plies=2,
            stockfish_gate_games=1,
            stockfish_gate_simulations=16,
            stockfish_gate_min_score=0.5,
            stockfish_gate_engine_path="tools/stockfish/bin/stockfish",
            device="cpu",
        )
    )

    league = json.loads(league_path.read_text())

    assert league["best_checkpoint"] == "parent.pt"
    assert league["history"][0]["promoted"] is False
    assert league["history"][0]["metrics"]["score_rate"] == pytest.approx(1.0)
    assert league["history"][0]["stockfish_gate_metrics"]["score_rate"] == pytest.approx(0.0)
    assert len(eval_calls) == 2
