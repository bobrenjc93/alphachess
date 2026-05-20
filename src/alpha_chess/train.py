"""Training loop for AlphaChess checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from alpha_chess.chess_env import ACTION_SIZE, legal_actions
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.model import ChessNet, ChessNetConfig, load_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    data: str | list[str]
    out: str = "checkpoints/run"
    checkpoint: str | None = None
    holdout_data: str | list[str] | None = None
    epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    bad_action_weight: float = 0.0
    bad_action_margin: float = 1.0
    data_weights: list[float] | None = None
    source_policy_weights: list[float] | None = None
    legal_policy_loss: bool = False
    color_mirror_augmentation: bool = False
    prefer_action_labels: bool = False
    policy_head_only: bool = False
    value_head_only: bool = False
    select_best_by: str | None = None
    channels: int = 128
    blocks: int = 6
    seed: int = 0
    device: str = "auto"


@dataclass
class ValidateConfig:
    checkpoint: str
    data: str | list[str]
    batch_size: int = 256
    value_weight: float = 1.0
    bad_action_weight: float = 0.0
    bad_action_margin: float = 1.0
    legal_policy_loss: bool = False
    prefer_action_labels: bool = False
    device: str = "auto"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    dataset = SelfPlayDataset(
        config.data,
        in_memory=True,
        color_mirror_augmentation=config.color_mirror_augmentation,
        prefer_action_labels=config.prefer_action_labels,
    )

    val_size = max(1, int(0.1 * len(dataset))) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size
    if val_size:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
    else:
        train_ds, val_ds = dataset, None

    train_sampler = _build_train_sampler(dataset, train_ds, config.data_weights, config.seed)
    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_samples,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_samples,
        )
        if val_ds is not None
        else None
    )
    holdout_dataset = (
        SelfPlayDataset(
            config.holdout_data,
            in_memory=True,
            prefer_action_labels=config.prefer_action_labels,
        )
        if config.holdout_data is not None
        else None
    )
    holdout_loader = (
        DataLoader(
            holdout_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_samples,
        )
        if holdout_dataset is not None
        else None
    )

    if config.checkpoint:
        model = load_checkpoint(config.checkpoint, map_location=device)
    else:
        model = ChessNet(ChessNetConfig(channels=config.channels, blocks=config.blocks))
    model.to(device)
    if config.policy_head_only and config.value_head_only:
        raise ValueError("policy_head_only and value_head_only are mutually exclusive")
    if config.policy_head_only:
        _freeze_except_policy_head(model)
    if config.value_head_only:
        _freeze_except_value_head(model)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are available")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    latest_metrics: dict[str, float] = {}
    best_metric_value: float | None = None

    for epoch in range(config.epochs):
        _set_train_mode(
            model,
            policy_head_only=config.policy_head_only,
            value_head_only=config.value_head_only,
        )
        running_loss = 0.0
        for batch in loader:
            loss, parts = _compute_batch_loss(
                model,
                batch,
                device,
                config.value_weight,
                legal_policy_loss=config.legal_policy_loss,
                bad_action_weight=config.bad_action_weight,
                bad_action_margin=config.bad_action_margin,
                source_policy_weights=config.source_policy_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            running_loss += float(loss.item())
            latest_metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(parts["policy_loss"].item()),
                "policy_acc": float(parts["policy_acc"].item()),
                "value_loss": float(parts["value_loss"].item()),
            }
            if "bad_action_loss" in parts:
                latest_metrics["bad_action_loss"] = float(parts["bad_action_loss"].item())

        latest_metrics["epoch_loss"] = running_loss / max(1, len(loader))
        if val_loader is not None:
            latest_metrics.update(
                _evaluate_loss(
                    model,
                    val_loader,
                    device,
                    config.value_weight,
                    legal_policy_loss=config.legal_policy_loss,
                    bad_action_weight=config.bad_action_weight,
                    bad_action_margin=config.bad_action_margin,
                    source_names=dataset.source_names if len(dataset.source_names) > 1 else None,
                )
            )
        if holdout_loader is not None and holdout_dataset is not None:
            latest_metrics.update(
                _evaluate_loss(
                    model,
                    holdout_loader,
                    device,
                    config.value_weight,
                    legal_policy_loss=config.legal_policy_loss,
                    bad_action_weight=config.bad_action_weight,
                    bad_action_margin=config.bad_action_margin,
                    source_names=(
                        holdout_dataset.source_names
                        if len(holdout_dataset.source_names) > 1
                        else None
                    ),
                    prefix="holdout",
                )
            )

        epoch_path = out_dir / f"epoch_{epoch + 1:04d}.pt"
        epoch_metrics = dict(latest_metrics)
        if config.select_best_by is not None:
            selected_value = _selected_metric_value(epoch_metrics, config.select_best_by)
            improved = _metric_improved(
                config.select_best_by,
                selected_value,
                best_metric_value,
            )
            epoch_metrics.update(
                {
                    "selected_by": config.select_best_by,
                    "selected_metric_value": selected_value,
                    "selected_as_latest": improved,
                }
            )
        else:
            improved = True

        save_checkpoint(epoch_path, model, optimizer, step, epoch_metrics)
        if config.select_best_by is None:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step, epoch_metrics)
        elif improved:
            best_metric_value = selected_value
            latest_metrics = dict(epoch_metrics)
            latest_metrics.update(
                {
                    "selected_epoch": epoch + 1,
                    "selected_checkpoint": epoch_path.name,
                }
            )
            save_checkpoint(out_dir / "latest.pt", model, optimizer, step, latest_metrics)

    return out_dir / "latest.pt"


def _selected_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(
            f"select_best_by metric {metric_name!r} was not produced; "
            f"available metrics: {available}"
        )
    value = float(metrics[metric_name])
    if not math.isfinite(value):
        raise ValueError(f"select_best_by metric {metric_name!r} is not finite: {value}")
    _metric_higher_is_better(metric_name)
    return value


def _metric_higher_is_better(metric_name: str) -> bool:
    if metric_name.endswith("_acc") or metric_name == "policy_acc":
        return True
    if metric_name.endswith("_loss") or metric_name in {"loss", "epoch_loss"}:
        return False
    raise ValueError(
        f"select_best_by metric {metric_name!r} is ambiguous; "
        "use a metric ending in _acc or _loss"
    )


def _metric_improved(
    metric_name: str,
    current_value: float,
    best_value: float | None,
) -> bool:
    if best_value is None:
        return True
    if _metric_higher_is_better(metric_name):
        return current_value > best_value
    return current_value < best_value


def _freeze_except_policy_head(model: ChessNet) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.policy_head.parameters():
        param.requires_grad_(True)


def _freeze_except_value_head(model: ChessNet) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.value_head.parameters():
        param.requires_grad_(True)


def _set_train_mode(
    model: ChessNet,
    policy_head_only: bool,
    value_head_only: bool,
) -> None:
    model.train()
    if not policy_head_only and not value_head_only:
        return
    model.stem.eval()
    model.blocks.eval()
    if policy_head_only:
        model.value_head.eval()
        model.policy_head.train()
    if value_head_only:
        model.policy_head.eval()
        model.value_head.train()


@torch.no_grad()
def validate(config: ValidateConfig) -> dict[str, float]:
    device = resolve_device(config.device)
    dataset = SelfPlayDataset(
        config.data,
        in_memory=True,
        prefer_action_labels=config.prefer_action_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    model = load_checkpoint(config.checkpoint, map_location=device)
    model.to(device)
    return _evaluate_loss(
        model,
        loader,
        device,
        config.value_weight,
        legal_policy_loss=config.legal_policy_loss,
        bad_action_weight=config.bad_action_weight,
        bad_action_margin=config.bad_action_margin,
        source_names=dataset.source_names if len(dataset.source_names) > 1 else None,
    )


def _build_train_sampler(
    dataset: SelfPlayDataset,
    train_ds: SelfPlayDataset | Subset,
    data_weights: list[float] | None,
    seed: int,
) -> WeightedRandomSampler | None:
    if data_weights is None:
        return None

    weights = dataset.source_sample_weights(data_weights)
    if isinstance(train_ds, Subset):
        weights = weights[list(train_ds.indices)]
    if float(weights.sum()) <= 0:
        raise ValueError("Weighted training split has no positive sample weights")

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def _evaluate_loss(
    model: ChessNet,
    loader: DataLoader,
    device: torch.device,
    value_weight: float,
    legal_policy_loss: bool = False,
    bad_action_weight: float = 0.0,
    bad_action_margin: float = 1.0,
    source_names: list[str] | None = None,
    prefix: str = "val",
) -> dict[str, float]:
    model.eval()
    totals = _new_eval_totals()
    source_totals = (
        {source_id: _new_eval_totals() for source_id in range(len(source_names))}
        if source_names is not None
        else {}
    )
    for batch in loader:
        loss, parts = _compute_batch_loss(
            model,
            batch,
            device,
            value_weight,
            legal_policy_loss=legal_policy_loss,
            bad_action_weight=bad_action_weight,
            bad_action_margin=bad_action_margin,
        )
        _add_eval_totals(totals, _batch_size(batch), loss, parts)

        if source_names is not None and "source_id" in batch:
            source_ids = _batch_tensor(batch, "source_id")
            for source_id_tensor in torch.unique(source_ids):
                source_id = int(source_id_tensor.item())
                if source_id not in source_totals:
                    continue
                mask = source_ids == source_id
                sub_batch = _slice_batch(batch, mask)
                sub_loss, sub_parts = _compute_batch_loss(
                    model,
                    sub_batch,
                    device,
                    value_weight,
                    legal_policy_loss=legal_policy_loss,
                    bad_action_weight=bad_action_weight,
                    bad_action_margin=bad_action_margin,
                )
                _add_eval_totals(
                    source_totals[source_id],
                    _batch_size(sub_batch),
                    sub_loss,
                    sub_parts,
                )

    metrics = _finalize_eval_totals(prefix, totals)
    for source_id, source_totals_for_id in source_totals.items():
        metrics.update(_finalize_eval_totals(f"{prefix}_source_{source_id}", source_totals_for_id))
    return metrics


def _new_eval_totals() -> dict[str, float]:
    return {
        "examples": 0.0,
        "loss": 0.0,
        "policy_loss": 0.0,
        "policy_acc": 0.0,
        "policy_top3_acc": 0.0,
        "policy_top5_acc": 0.0,
        "value_loss": 0.0,
        "bad_action_loss": 0.0,
    }


def _add_eval_totals(
    totals: dict[str, float],
    examples: int,
    loss: torch.Tensor,
    parts: dict[str, torch.Tensor],
) -> None:
    totals["examples"] += examples
    totals["loss"] += float(loss.item()) * examples
    totals["policy_loss"] += float(parts["policy_loss"].item()) * examples
    totals["policy_acc"] += float(parts["policy_acc"].item()) * examples
    if "policy_top3_acc" in parts:
        totals["policy_top3_acc"] += float(parts["policy_top3_acc"].item()) * examples
    if "policy_top5_acc" in parts:
        totals["policy_top5_acc"] += float(parts["policy_top5_acc"].item()) * examples
    totals["value_loss"] += float(parts["value_loss"].item()) * examples
    if "bad_action_loss" in parts:
        totals["bad_action_loss"] += float(parts["bad_action_loss"].item()) * examples


def _finalize_eval_totals(prefix: str, totals: dict[str, float]) -> dict[str, float]:
    examples = totals["examples"]
    if examples <= 0:
        return {f"{prefix}_examples": 0.0}
    return {
        f"{prefix}_loss": totals["loss"] / examples,
        f"{prefix}_policy_loss": totals["policy_loss"] / examples,
        f"{prefix}_policy_acc": totals["policy_acc"] / examples,
        f"{prefix}_policy_top3_acc": totals["policy_top3_acc"] / examples,
        f"{prefix}_policy_top5_acc": totals["policy_top5_acc"] / examples,
        f"{prefix}_value_loss": totals["value_loss"] / examples,
        f"{prefix}_bad_action_loss": totals["bad_action_loss"] / examples,
        f"{prefix}_examples": examples,
    }


def _compute_batch_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    value_weight: float,
    legal_policy_loss: bool = False,
    bad_action_weight: float = 0.0,
    bad_action_margin: float = 1.0,
    source_policy_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boards = _batch_tensor(batch, "board").to(device)
    values = _batch_tensor(batch, "value").to(device)
    if legal_policy_loss:
        return _compute_legal_masked_loss(
            model,
            batch,
            boards,
            values,
            device,
            value_weight,
            bad_action_weight,
            bad_action_margin,
            source_policy_weights=source_policy_weights,
        )
    return _compute_unmasked_loss(
        model,
        batch,
        boards,
        values,
        device,
        value_weight,
        bad_action_weight,
        bad_action_margin,
        source_policy_weights=source_policy_weights,
    )


def _compute_legal_masked_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor | list[str]],
    boards: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    value_weight: float,
    bad_action_weight: float = 0.0,
    bad_action_margin: float = 1.0,
    source_policy_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "fen" not in batch:
        raise ValueError("legal_policy_loss requires training NPZ files with stored FENs")
    fens = batch["fen"]
    if not isinstance(fens, list):
        raise TypeError("Batch FENs must be a list of strings")

    policy_logits, value = model(boards)
    legal_mask = _legal_action_mask_from_fens(fens, device)
    masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)

    if "policy" in batch:
        target_policy = _batch_tensor(batch, "policy").to(device)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_example_policy_loss = -(target_policy * log_probs).sum(dim=-1)
        target_action = target_policy.argmax(dim=-1)
    else:
        target_action = _batch_tensor(batch, "action").to(device).long()
        per_example_policy_loss = F.cross_entropy(masked_logits, target_action, reduction="none")

    target_is_legal = legal_mask.gather(1, target_action.unsqueeze(1)).squeeze(1)
    if not bool(target_is_legal.all().item()):
        raise ValueError("legal_policy_loss received a target action that is illegal for its FEN")

    policy_weight = _source_weight_vector(
        batch,
        source_policy_weights,
        device,
        "source_policy_weights",
    )
    policy_loss = _weighted_mean(per_example_policy_loss, policy_weight)
    policy_acc = (masked_logits.argmax(dim=-1) == target_action).float().mean()
    value_loss = F.mse_loss(value, values)
    bad_action_loss = _bad_action_margin_loss(
        masked_logits,
        target_action,
        batch,
        device,
        margin=bad_action_margin,
    )
    loss = policy_loss + value_weight * value_loss + bad_action_weight * bad_action_loss
    parts = {
        "policy_loss": policy_loss.detach(),
        "policy_acc": policy_acc.detach(),
        "value_loss": value_loss.detach(),
    }
    parts.update(_policy_topk_metrics(masked_logits, target_action))
    if bad_action_weight > 0 and "bad_action" in batch:
        parts["bad_action_loss"] = bad_action_loss.detach()
    return loss, parts


def _compute_unmasked_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor | list[str]],
    boards: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    value_weight: float,
    bad_action_weight: float,
    bad_action_margin: float,
    source_policy_weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_logits, value = model(boards)
    if "policy" in batch:
        target_policy = _batch_tensor(batch, "policy").to(device)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        per_example_policy_loss = -(target_policy * log_probs).sum(dim=-1)
        target_action = target_policy.argmax(dim=-1)
    else:
        target_action = _batch_tensor(batch, "action").to(device).long()
        per_example_policy_loss = F.cross_entropy(policy_logits, target_action, reduction="none")

    policy_weight = _source_weight_vector(
        batch,
        source_policy_weights,
        device,
        "source_policy_weights",
    )
    policy_loss = _weighted_mean(per_example_policy_loss, policy_weight)
    policy_acc = (policy_logits.argmax(dim=-1) == target_action).float().mean()
    value_loss = F.mse_loss(value, values)
    bad_action_loss = _bad_action_margin_loss(
        policy_logits,
        target_action,
        batch,
        device,
        margin=bad_action_margin,
    )
    loss = policy_loss + value_weight * value_loss + bad_action_weight * bad_action_loss
    parts = {
        "policy_loss": policy_loss.detach(),
        "policy_acc": policy_acc.detach(),
        "value_loss": value_loss.detach(),
    }
    parts.update(_policy_topk_metrics(policy_logits, target_action))
    if bad_action_weight > 0 and "bad_action" in batch:
        parts["bad_action_loss"] = bad_action_loss.detach()
    return loss, parts


def _policy_topk_metrics(
    policy_logits: torch.Tensor,
    target_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    width = int(policy_logits.shape[1])
    metrics: dict[str, torch.Tensor] = {}
    for k in (3, 5):
        effective_k = min(k, width)
        topk = torch.topk(policy_logits, k=effective_k, dim=-1).indices
        hit = (topk == target_action.long().unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"policy_top{k}_acc"] = hit.detach()
    return metrics


def _bad_action_margin_loss(
    policy_logits: torch.Tensor,
    target_action: torch.Tensor,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    if "bad_action" not in batch:
        return policy_logits.new_zeros(())

    bad_action = _batch_tensor(batch, "bad_action").to(device).long()
    valid = (bad_action >= 0) & (bad_action != target_action)
    if not bool(valid.any().item()):
        return policy_logits.new_zeros(())

    target_logits = policy_logits.gather(1, target_action.unsqueeze(1)).squeeze(1)
    bad_logits = policy_logits.gather(1, bad_action.clamp_min(0).unsqueeze(1)).squeeze(1)
    return F.softplus(bad_logits[valid] - target_logits[valid] + margin).mean()


def _source_weight_vector(
    batch: dict[str, torch.Tensor | list[str]],
    weights: list[float] | None,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if "source_id" not in batch:
        raise ValueError(f"{name} requires batches with source_id")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError(f"{name} must be finite and non-negative")

    source_ids = _batch_tensor(batch, "source_id").to(device).long()
    if source_ids.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    max_source_id = int(source_ids.max().item())
    if max_source_id >= len(weights):
        raise ValueError(f"{name} has no entry for source id {max_source_id}")

    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return weight_tensor[source_ids]


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return losses.mean()
    return (losses * weights).mean()


def _legal_action_mask_from_fens(fens: list[str], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(fens), ACTION_SIZE), dtype=torch.bool, device=device)
    for row, fen in enumerate(fens):
        actions = legal_actions(chess.Board(fen))
        if actions:
            mask[row, torch.as_tensor(actions, dtype=torch.long, device=device)] = True
    return mask


def _batch_size(batch: dict[str, torch.Tensor | list[str]]) -> int:
    return int(_batch_tensor(batch, "board").shape[0])


def _slice_batch(
    batch: dict[str, torch.Tensor | list[str]],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor | list[str]]:
    indices = mask.nonzero(as_tuple=False).flatten()
    sliced: dict[str, torch.Tensor | list[str]] = {}
    index_list = [int(index) for index in indices.tolist()]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[indices]
        else:
            sliced[key] = [value[index] for index in index_list]
    return sliced


def _batch_tensor(batch: dict[str, torch.Tensor | list[str]], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key} must be a tensor")
    return value
