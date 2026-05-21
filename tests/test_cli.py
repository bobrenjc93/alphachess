from pathlib import Path

import alpha_chess.__main__ as cli


def test_hard_negatives_cli_prefers_action_labels_by_default(monkeypatch, tmp_path):
    captured = {}

    def fake_mine(config):
        captured["config"] = config
        return [tmp_path / "hard_negatives.npz"]

    monkeypatch.setattr(cli, "mine_hard_negatives", fake_mine)

    cli.main(
        [
            "hard-negatives",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "teacher-data",
            "--out",
            str(tmp_path),
        ]
    )

    assert captured["config"].prefer_action_labels is True


def test_model_blunders_cli_prefers_action_labels_by_default(monkeypatch, tmp_path):
    captured = {}

    def fake_mine(config):
        captured["config"] = config
        return [Path(tmp_path) / "model_blunders.npz"]

    monkeypatch.setattr(cli, "mine_model_blunders", fake_mine)

    cli.main(
        [
            "model-blunders",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "teacher-data",
            "--out",
            str(tmp_path),
        ]
    )

    assert captured["config"].prefer_action_labels is True


def test_model_blunders_cli_can_disable_action_label_preference(monkeypatch, tmp_path):
    captured = {}

    def fake_mine(config):
        captured["config"] = config
        return [Path(tmp_path) / "model_blunders.npz"]

    monkeypatch.setattr(cli, "mine_model_blunders", fake_mine)

    cli.main(
        [
            "model-blunders",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "teacher-data",
            "--out",
            str(tmp_path),
            "--no-prefer-action-labels",
        ]
    )

    assert captured["config"].prefer_action_labels is False
