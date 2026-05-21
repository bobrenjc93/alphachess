import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_chess.chess_env import (
    ACTION_SIZE,
    NUM_INPUT_PLANES,
    action_to_move,
    color_mirror_board,
    color_mirror_move,
    encode_board,
    move_to_action,
)
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.model import ChessNet, ChessNetConfig, blend_checkpoints, save_checkpoint
from alpha_chess.train import (
    TrainConfig,
    ValidateConfig,
    _compute_batch_loss,
    _evaluate_loss,
    _legal_action_mask_from_fens,
    _metric_improved,
    train,
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


def test_source_sample_weights_can_cap_tiny_source_repeats(tmp_path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_sparse_npz(source_a / "a.npz", positions=100)
    _write_sparse_npz(source_b / "b.npz", positions=2)

    dataset = SelfPlayDataset([source_a, source_b])
    weights = dataset.source_sample_weights(
        [0.5, 0.5],
        max_source_repeat=5.0,
        num_samples=len(dataset),
    )
    probabilities = weights / weights.sum()

    assert torch.isclose(
        probabilities[100:].sum(),
        torch.tensor(10 / 102, dtype=torch.double),
    )
    assert torch.isclose(probabilities[:100].sum(), torch.tensor(92 / 102, dtype=torch.double))

    with pytest.raises(ValueError, match="too low"):
        dataset.source_sample_weights(
            [0.5, 0.5],
            max_source_repeat=0.5,
            num_samples=len(dataset),
        )


def test_dataset_collates_fens_for_legal_policy_loss(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    bad_move = chess.Move.from_uci("d2d4")
    np.savez_compressed(
        tmp_path / "fen.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        actions=np.asarray([move_to_action(move, board)], dtype=np.int64),
        bad_actions=np.asarray([move_to_action(bad_move, board)], dtype=np.int64),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    dataset = SelfPlayDataset(tmp_path, in_memory=True)
    sample = dataset[0]
    batch = collate_samples([sample])

    assert sample["fen"] == board.fen()
    assert int(sample["bad_action"]) == move_to_action(bad_move, board)
    assert batch["fen"] == [board.fen()]
    assert int(batch["bad_action"][0]) == move_to_action(bad_move, board)


def test_dataset_collates_multiple_bad_actions(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    bad_moves = [chess.Move.from_uci("d2d4"), chess.Move.from_uci("g1f3")]
    bad_actions = [move_to_action(bad_move, board) for bad_move in bad_moves]
    np.savez_compressed(
        tmp_path / "fen.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        actions=np.asarray([move_to_action(move, board)], dtype=np.int64),
        bad_actions=np.asarray([bad_actions], dtype=np.int64),
        bad_action_deltas=np.asarray([[0.25, 0.75]], dtype=np.float32),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    sample = SelfPlayDataset(tmp_path, in_memory=True)[0]
    batch = collate_samples([sample])

    assert sample["bad_action"].tolist() == bad_actions
    assert sample["bad_action_delta"].tolist() == pytest.approx([0.25, 0.75])
    assert batch["bad_action"].shape == (1, 2)
    assert batch["bad_action"][0].tolist() == bad_actions
    assert batch["bad_action_delta"].shape == (1, 2)
    assert batch["bad_action_delta"][0].tolist() == pytest.approx([0.25, 0.75])


def test_dataset_color_mirror_augmentation_maps_labels(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    bad_move = chess.Move.from_uci("d2d4")
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[move_to_action(move, board)] = 0.75
    policy[move_to_action(bad_move, board)] = 0.25
    np.savez_compressed(
        tmp_path / "fen.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        policies=np.asarray([policy], dtype=np.float32),
        actions=np.asarray([move_to_action(move, board)], dtype=np.int64),
        bad_actions=np.asarray([move_to_action(bad_move, board)], dtype=np.int64),
        values=np.asarray([0.5], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    dataset = SelfPlayDataset(tmp_path, in_memory=True, color_mirror_augmentation=True)
    mirrored = dataset[1]
    mirrored_board = color_mirror_board(board)
    mirrored_move = color_mirror_move(move)
    mirrored_bad_move = color_mirror_move(bad_move)
    mirrored_action = move_to_action(mirrored_move, mirrored_board)
    mirrored_bad_action = move_to_action(mirrored_bad_move, mirrored_board)

    assert len(dataset) == 2
    assert mirrored["fen"] == mirrored_board.fen()
    assert int(mirrored["action"]) == mirrored_action
    assert int(mirrored["bad_action"]) == mirrored_bad_action
    assert action_to_move(int(mirrored["action"]), mirrored_board) == mirrored_move
    assert float(mirrored["value"]) == pytest.approx(0.5)
    assert float(mirrored["policy"][mirrored_action]) == pytest.approx(0.75)
    assert float(mirrored["policy"][mirrored_bad_action]) == pytest.approx(0.25)


def test_dataset_can_prefer_actions_over_dense_policy(tmp_path) -> None:
    board = chess.Board()
    best_move = chess.Move.from_uci("e2e4")
    soft_move = chess.Move.from_uci("d2d4")
    best_action = move_to_action(best_move, board)
    soft_action = move_to_action(soft_move, board)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[soft_action] = 1.0
    np.savez_compressed(
        tmp_path / "teacher.npz",
        boards=np.asarray([encode_board(board)], dtype=np.float32),
        policies=np.asarray([policy], dtype=np.float32),
        actions=np.asarray([best_action], dtype=np.int64),
        values=np.asarray([0.0], dtype=np.float32),
        fens=np.asarray([board.fen()]),
    )

    default_sample = SelfPlayDataset(tmp_path, in_memory=True)[0]
    preferred_sample = SelfPlayDataset(
        tmp_path,
        in_memory=True,
        prefer_action_labels=True,
    )[0]
    preferred_batch = collate_samples([preferred_sample])

    assert "policy" in default_sample
    assert int(default_sample["action"]) == best_action
    assert int(default_sample["policy"].argmax()) == soft_action
    assert "policy" not in preferred_sample
    assert int(preferred_sample["action"]) == best_action
    assert "action" in preferred_batch
    assert int(preferred_batch["action"][0]) == best_action


def test_color_mirror_source_weights_duplicate_base_weights(tmp_path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    _write_sparse_npz(source_a / "a.npz", positions=2, include_fens=True)
    _write_sparse_npz(source_b / "b.npz", positions=6, include_fens=True)

    dataset = SelfPlayDataset([source_a, source_b], color_mirror_augmentation=True)
    weights = dataset.source_sample_weights([0.75, 0.25])

    assert len(dataset) == 16
    assert torch.allclose(weights[:8], weights[8:])
    assert torch.isclose(weights[:2].sum(), torch.tensor(0.75, dtype=torch.double))
    assert torch.isclose(weights[8:10].sum(), torch.tensor(0.75, dtype=torch.double))


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
    assert "policy_top3_acc" in parts
    assert "policy_top5_acc" in parts


def test_bad_action_margin_loss_is_reported() -> None:
    board = chess.Board()
    target = move_to_action(chess.Move.from_uci("e2e4"), board)
    bad = move_to_action(chess.Move.from_uci("d2d4"), board)
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    batch = {
        "board": torch.from_numpy(np.asarray([encode_board(board)], dtype=np.float32)),
        "action": torch.tensor([target], dtype=torch.long),
        "bad_action": torch.tensor([bad], dtype=torch.long),
        "value": torch.zeros(1),
        "fen": [board.fen()],
    }

    loss, parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=1.0,
        legal_policy_loss=True,
        bad_action_weight=0.5,
    )

    assert loss.ndim == 0
    assert "bad_action_loss" in parts
    assert float(parts["bad_action_loss"]) >= 0.0


def test_bad_action_margin_loss_accepts_multiple_bad_actions() -> None:
    board = chess.Board()
    target = move_to_action(chess.Move.from_uci("e2e4"), board)
    bad_actions = [
        move_to_action(chess.Move.from_uci("d2d4"), board),
        move_to_action(chess.Move.from_uci("g1f3"), board),
    ]
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    batch = {
        "board": torch.from_numpy(np.asarray([encode_board(board)], dtype=np.float32)),
        "action": torch.tensor([target], dtype=torch.long),
        "bad_action": torch.tensor([bad_actions], dtype=torch.long),
        "value": torch.zeros(1),
        "fen": [board.fen()],
    }

    loss, parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=1.0,
        legal_policy_loss=True,
        bad_action_weight=0.5,
    )

    assert loss.ndim == 0
    assert "bad_action_loss" in parts
    assert float(parts["bad_action_loss"]) >= 0.0


def test_bad_action_margin_loss_can_weight_value_deltas() -> None:
    target = 10
    bad_actions = [11, 12]

    class FixedPolicy(torch.nn.Module):
        def forward(self, boards):
            logits = torch.full((boards.shape[0], ACTION_SIZE), -10.0)
            logits[:, target] = 0.0
            logits[:, bad_actions[0]] = 0.0
            logits[:, bad_actions[1]] = 2.0
            return logits, torch.zeros(boards.shape[0])

    batch = {
        "board": torch.zeros(1, NUM_INPUT_PLANES, 8, 8),
        "action": torch.tensor([target], dtype=torch.long),
        "bad_action": torch.tensor([bad_actions], dtype=torch.long),
        "bad_action_delta": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        "value": torch.zeros(1),
    }

    _plain_loss, plain_parts = _compute_batch_loss(
        FixedPolicy(),
        batch,
        torch.device("cpu"),
        value_weight=0.0,
        bad_action_weight=1.0,
        bad_action_margin=1.0,
    )
    _weighted_loss, weighted_parts = _compute_batch_loss(
        FixedPolicy(),
        batch,
        torch.device("cpu"),
        value_weight=0.0,
        bad_action_weight=1.0,
        bad_action_margin=1.0,
        bad_action_delta_weight=1.0,
    )

    assert float(weighted_parts["bad_action_loss"]) > float(
        plain_parts["bad_action_loss"]
    )


def test_policy_distillation_loss_anchors_to_reference_model() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    teacher = ChessNet(ChessNetConfig(channels=8, blocks=1))
    for param in teacher.parameters():
        param.data.zero_()
    teacher.eval()
    batch = {
        "board": torch.from_numpy(np.stack([encode_board(board)])).float(),
        "action": torch.tensor([move_to_action(move, board)], dtype=torch.long),
        "value": torch.tensor([0.0], dtype=torch.float32),
        "fen": [board.fen()],
    }

    plain_loss, _plain_parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=0.0,
        legal_policy_loss=True,
    )
    anchored_loss, anchored_parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=0.0,
        legal_policy_loss=True,
        distill_model=teacher,
        policy_distill_weight=0.5,
    )

    assert "policy_distill_loss" in anchored_parts
    assert anchored_parts["policy_distill_loss"].item() > 0.0
    assert anchored_loss.detach().item() > plain_loss.detach().item()


def test_train_uses_separate_distill_data(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    distill_dir = tmp_path / "distill"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    distill_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=4)
    _write_sparse_npz(distill_dir / "anchor.npz", positions=5)
    teacher_path = tmp_path / "teacher.pt"
    save_checkpoint(teacher_path, ChessNet(ChessNetConfig(channels=8, blocks=1)))
    distill_batch_sizes = []

    def fake_distill_only_loss(*args, **_kwargs):
        batch = args[1]
        distill_batch_sizes.append(int(batch["board"].shape[0]))
        return torch.as_tensor(0.25)

    monkeypatch.setattr(
        "alpha_chess.train._compute_policy_distill_only_loss",
        fake_distill_only_loss,
    )

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            distill_checkpoint=str(teacher_path),
            distill_data=str(distill_dir),
            distill_batch_size=3,
            policy_distill_weight=0.5,
            epochs=1,
            batch_size=2,
            channels=8,
            blocks=1,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            device="cpu",
        )
    )

    latest = torch.load(out_dir / "latest.pt", map_location="cpu")
    assert distill_batch_sizes
    assert distill_batch_sizes[0] == 3
    assert latest["metrics"]["policy_distill_loss"] == pytest.approx(0.25)


def test_source_policy_weights_scale_policy_loss() -> None:
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    model.eval()
    boards = torch.zeros(2, NUM_INPUT_PLANES, 8, 8)
    actions = torch.tensor([0, 1], dtype=torch.long)
    batch = {
        "board": boards,
        "action": actions,
        "value": torch.zeros(2),
        "source_id": torch.tensor([0, 1], dtype=torch.long),
    }

    policy_logits, _value = model(boards)
    expected_policy_loss = F.cross_entropy(policy_logits, actions, reduction="none")[1] / 2
    loss, parts = _compute_batch_loss(
        model,
        batch,
        torch.device("cpu"),
        value_weight=0.0,
        source_policy_weights=[0.0, 1.0],
    )

    assert loss.item() == pytest.approx(float(expected_policy_loss.detach()))
    assert parts["policy_loss"].item() == pytest.approx(float(expected_policy_loss.detach()))


def test_source_policy_weights_validate_source_count() -> None:
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    model.eval()
    batch = {
        "board": torch.zeros(1, NUM_INPUT_PLANES, 8, 8),
        "action": torch.tensor([0], dtype=torch.long),
        "value": torch.zeros(1),
        "source_id": torch.tensor([1], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="source_policy_weights"):
        _compute_batch_loss(
            model,
            batch,
            torch.device("cpu"),
            value_weight=0.0,
            source_policy_weights=[1.0],
        )


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
    assert "val_source_0_policy_top3_acc" in metrics
    assert "val_source_0_policy_top5_acc" in metrics
    assert "val_source_1_policy_acc" in metrics
    assert "val_source_1_policy_top3_acc" in metrics
    assert "val_source_1_policy_top5_acc" in metrics

    holdout_metrics = _evaluate_loss(
        model,
        loader,
        torch.device("cpu"),
        value_weight=1.0,
        source_names=dataset.source_names,
        prefix="holdout",
    )

    assert holdout_metrics["holdout_examples"] == 8.0
    assert holdout_metrics["holdout_source_0_examples"] == 3.0
    assert holdout_metrics["holdout_source_1_examples"] == 5.0
    assert "holdout_source_0_policy_acc" in holdout_metrics


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
    assert "val_policy_top3_acc" in metrics
    assert "val_policy_top5_acc" in metrics


def test_metric_improved_uses_loss_and_accuracy_direction() -> None:
    assert _metric_improved("val_policy_acc", 0.7, 0.6)
    assert not _metric_improved("val_policy_acc", 0.6, 0.7)
    assert _metric_improved("val_policy_loss", 1.2, 1.3)
    assert not _metric_improved("val_policy_loss", 1.4, 1.3)

    with pytest.raises(ValueError, match="ambiguous"):
        _metric_improved("val_examples", 2.0, 1.0)


def test_train_select_best_by_keeps_best_validation_epoch(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=12)

    eval_calls = []

    def fake_evaluate_loss(*_args, **_kwargs):
        eval_calls.append(None)
        if len(eval_calls) == 1:
            return {"val_loss": 2.0, "val_policy_acc": 0.9}
        return {"val_loss": 1.0, "val_policy_acc": 0.1}

    monkeypatch.setattr("alpha_chess.train._evaluate_loss", fake_evaluate_loss)

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            epochs=2,
            batch_size=4,
            channels=8,
            blocks=1,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            select_best_by="val_policy_acc",
            device="cpu",
        )
    )

    assert len(eval_calls) == 2
    latest = torch.load(out_dir / "latest.pt", map_location="cpu")
    epoch1 = torch.load(out_dir / "epoch_0001.pt", map_location="cpu")
    epoch2 = torch.load(out_dir / "epoch_0002.pt", map_location="cpu")

    assert latest["metrics"]["selected_by"] == "val_policy_acc"
    assert latest["metrics"]["selected_metric_value"] == pytest.approx(0.9)
    assert latest["metrics"]["selected_epoch"] == 1
    assert latest["metrics"]["selected_checkpoint"] == "epoch_0001.pt"

    for key, tensor in epoch1["model_state"].items():
        assert torch.equal(latest["model_state"][key], tensor), key

    assert any(
        not torch.equal(latest["model_state"][key], tensor)
        for key, tensor in epoch2["model_state"].items()
        if torch.is_floating_point(tensor)
    )


def test_train_can_select_best_by_holdout_metric(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    holdout_dir = tmp_path / "holdout"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    holdout_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=8)
    _write_sparse_npz(holdout_dir / "holdout.npz", positions=4)

    holdout_scores = [0.1, 0.8]
    prefixes = []

    def fake_evaluate_loss(*_args, **kwargs):
        prefixes.append(kwargs.get("prefix", "val"))
        score = holdout_scores[len(prefixes) - 1]
        return {
            "holdout_loss": 1.0 - score,
            "holdout_policy_acc": score,
            "holdout_examples": 4.0,
        }

    monkeypatch.setattr("alpha_chess.train._evaluate_loss", fake_evaluate_loss)

    train(
        TrainConfig(
            data=str(data_dir),
            holdout_data=str(holdout_dir),
            out=str(out_dir),
            epochs=2,
            batch_size=4,
            channels=8,
            blocks=1,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            select_best_by="holdout_policy_acc",
            device="cpu",
        )
    )

    latest = torch.load(out_dir / "latest.pt", map_location="cpu")

    assert prefixes == ["holdout", "holdout"]
    assert latest["metrics"]["selected_by"] == "holdout_policy_acc"
    assert latest["metrics"]["selected_metric_value"] == pytest.approx(0.8)
    assert latest["metrics"]["selected_epoch"] == 2
    assert latest["metrics"]["selected_checkpoint"] == "epoch_0002.pt"


def test_train_can_select_best_by_composite_metric(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=12)

    eval_metrics = [
        {"val_loss": 1.0, "val_policy_acc": 0.30, "val_policy_top3_acc": 0.85},
        {"val_loss": 1.0, "val_policy_acc": 0.55, "val_policy_top3_acc": 0.45},
    ]

    def fake_evaluate_loss(*_args, **_kwargs):
        return eval_metrics.pop(0)

    monkeypatch.setattr("alpha_chess.train._evaluate_loss", fake_evaluate_loss)

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            epochs=2,
            batch_size=4,
            channels=8,
            blocks=1,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            select_best_by="val_policy_acc+val_policy_top3_acc",
            device="cpu",
        )
    )

    latest = torch.load(out_dir / "latest.pt", map_location="cpu")

    assert latest["metrics"]["selected_by"] == "val_policy_acc+val_policy_top3_acc"
    assert latest["metrics"]["selected_metric_value"] == pytest.approx(1.15)
    assert latest["metrics"]["selected_epoch"] == 1
    assert latest["metrics"]["selected_checkpoint"] == "epoch_0001.pt"


def test_train_select_best_require_skips_ineligible_epoch(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=12)

    eval_metrics = [
        {"val_loss": 1.0, "val_policy_acc": 0.40, "val_policy_top3_acc": 0.70},
        {"val_loss": 1.0, "val_policy_acc": 0.90, "val_policy_top3_acc": 0.60},
    ]

    def fake_evaluate_loss(*_args, **_kwargs):
        return eval_metrics.pop(0)

    monkeypatch.setattr("alpha_chess.train._evaluate_loss", fake_evaluate_loss)

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            epochs=2,
            batch_size=4,
            channels=8,
            blocks=1,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            select_best_by="val_policy_acc",
            select_best_require=["val_policy_top3_acc>=0.65"],
            device="cpu",
        )
    )

    latest = torch.load(out_dir / "latest.pt", map_location="cpu")
    epoch2 = torch.load(out_dir / "epoch_0002.pt", map_location="cpu")

    assert latest["metrics"]["selected_metric_value"] == pytest.approx(0.40)
    assert latest["metrics"]["selected_epoch"] == 1
    assert latest["metrics"]["selected_requirements"] == "val_policy_top3_acc>=0.65"
    assert epoch2["metrics"]["selected_metric_value"] == pytest.approx(0.90)
    assert epoch2["metrics"]["selected_eligible"] is False
    assert epoch2["metrics"]["selected_as_latest"] is False


def test_train_select_best_by_rejects_missing_metric(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=12)

    monkeypatch.setattr(
        "alpha_chess.train._evaluate_loss",
        lambda *_args, **_kwargs: {"val_loss": 1.0},
    )

    with pytest.raises(ValueError, match="val_policy_acc"):
        train(
            TrainConfig(
                data=str(data_dir),
                out=str(out_dir),
                epochs=1,
                batch_size=4,
                channels=8,
                blocks=1,
                select_best_by="val_policy_acc",
                device="cpu",
            )
        )


def test_train_policy_head_only_freezes_body_and_value_head(tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=8)

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, ChessNet(ChessNetConfig(channels=8, blocks=1)))
    before = torch.load(checkpoint, map_location="cpu")["model_state"]

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            checkpoint=str(checkpoint),
            epochs=1,
            batch_size=4,
            lr=0.01,
            value_weight=0.0,
            weight_decay=0.0,
            policy_head_only=True,
            device="cpu",
        )
    )

    after = torch.load(out_dir / "latest.pt", map_location="cpu")["model_state"]
    changed_policy_keys = []
    for key, before_tensor in before.items():
        after_tensor = after[key]
        if key.startswith("policy_head."):
            if torch.is_floating_point(before_tensor) and not torch.allclose(
                before_tensor,
                after_tensor,
            ):
                changed_policy_keys.append(key)
            continue
        assert torch.equal(after_tensor, before_tensor), key

    assert changed_policy_keys


def test_train_value_head_only_freezes_body_and_policy_head(tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=8)

    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, ChessNet(ChessNetConfig(channels=8, blocks=1)))
    before = torch.load(checkpoint, map_location="cpu")["model_state"]

    train(
        TrainConfig(
            data=str(data_dir),
            out=str(out_dir),
            checkpoint=str(checkpoint),
            epochs=1,
            batch_size=4,
            lr=0.01,
            value_weight=1.0,
            weight_decay=0.0,
            value_head_only=True,
            device="cpu",
        )
    )

    after = torch.load(out_dir / "latest.pt", map_location="cpu")["model_state"]
    changed_value_keys = []
    for key, before_tensor in before.items():
        after_tensor = after[key]
        if key.startswith("value_head."):
            if torch.is_floating_point(before_tensor) and not torch.allclose(
                before_tensor,
                after_tensor,
            ):
                changed_value_keys.append(key)
            continue
        assert torch.equal(after_tensor, before_tensor), key

    assert changed_value_keys


def test_train_rejects_policy_and_value_head_only_together(tmp_path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_sparse_npz(data_dir / "game.npz", positions=1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        train(
            TrainConfig(
                data=str(data_dir),
                out=str(out_dir),
                policy_head_only=True,
                value_head_only=True,
                device="cpu",
            )
        )


def test_blend_checkpoints_interpolates_floating_weights(tmp_path) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    output = tmp_path / "blend.pt"

    model_a = _constant_model(0.0)
    model_b = _constant_model(2.0)
    save_checkpoint(checkpoint_a, model_a)
    save_checkpoint(checkpoint_b, model_b)

    blend_checkpoints(checkpoint_a, checkpoint_b, output, weight_b=0.25)

    payload = torch.load(output, map_location="cpu")
    state = payload["model_state"]
    floating_key = next(key for key, value in state.items() if torch.is_floating_point(value))

    assert torch.allclose(state[floating_key], torch.full_like(state[floating_key], 0.5))
    assert payload["metrics"]["blend_weight_b"] == pytest.approx(0.25)


def _constant_model(value: float) -> ChessNet:
    model = ChessNet(ChessNetConfig(channels=8, blocks=1))
    state = model.state_dict()
    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            state[key] = torch.full_like(tensor, value)
    model.load_state_dict(state)
    return model


def _write_sparse_npz(path, positions: int, include_fens: bool = False) -> None:
    boards = np.zeros((positions, NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    actions = np.zeros((positions,), dtype=np.int64)
    values = np.zeros((positions,), dtype=np.float32)
    payload = {"boards": boards, "actions": actions, "values": values}
    if include_fens:
        payload["fens"] = np.asarray([chess.Board().fen()] * positions)
    np.savez_compressed(path, **payload)
