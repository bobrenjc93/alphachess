"""Training loop for AlphaChess checkpoints."""

from __future__ import annotations

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
    epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    data_weights: list[float] | None = None
    legal_policy_loss: bool = False
    channels: int = 128
    blocks: int = 6
    seed: int = 0
    device: str = "auto"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    dataset = SelfPlayDataset(config.data, in_memory=True)

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

    if config.checkpoint:
        model = load_checkpoint(config.checkpoint, map_location=device)
    else:
        model = ChessNet(ChessNetConfig(channels=config.channels, blocks=config.blocks))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    latest_metrics: dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            loss, parts = _compute_batch_loss(
                model,
                batch,
                device,
                config.value_weight,
                legal_policy_loss=config.legal_policy_loss,
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

        latest_metrics["epoch_loss"] = running_loss / max(1, len(loader))
        if val_loader is not None:
            latest_metrics.update(
                _evaluate_loss(
                    model,
                    val_loader,
                    device,
                    config.value_weight,
                    legal_policy_loss=config.legal_policy_loss,
                    source_names=dataset.source_names if len(dataset.source_names) > 1 else None,
                )
            )

        save_checkpoint(out_dir / f"epoch_{epoch + 1:04d}.pt", model, optimizer, step, latest_metrics)
        save_checkpoint(out_dir / "latest.pt", model, optimizer, step, latest_metrics)

    return out_dir / "latest.pt"


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
    source_names: list[str] | None = None,
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
                )
                _add_eval_totals(
                    source_totals[source_id],
                    _batch_size(sub_batch),
                    sub_loss,
                    sub_parts,
                )

    metrics = _finalize_eval_totals("val", totals)
    for source_id, source_totals_for_id in source_totals.items():
        metrics.update(_finalize_eval_totals(f"val_source_{source_id}", source_totals_for_id))
    return metrics


def _new_eval_totals() -> dict[str, float]:
    return {
        "examples": 0.0,
        "loss": 0.0,
        "policy_loss": 0.0,
        "policy_acc": 0.0,
        "value_loss": 0.0,
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
    totals["value_loss"] += float(parts["value_loss"].item()) * examples


def _finalize_eval_totals(prefix: str, totals: dict[str, float]) -> dict[str, float]:
    examples = totals["examples"]
    if examples <= 0:
        return {f"{prefix}_examples": 0.0}
    return {
        f"{prefix}_loss": totals["loss"] / examples,
        f"{prefix}_policy_loss": totals["policy_loss"] / examples,
        f"{prefix}_policy_acc": totals["policy_acc"] / examples,
        f"{prefix}_value_loss": totals["value_loss"] / examples,
        f"{prefix}_examples": examples,
    }


def _compute_batch_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    value_weight: float,
    legal_policy_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boards = _batch_tensor(batch, "board").to(device)
    values = _batch_tensor(batch, "value").to(device)
    if legal_policy_loss:
        return _compute_legal_masked_loss(model, batch, boards, values, device, value_weight)
    if "policy" in batch:
        return model.compute_loss(
            boards,
            _batch_tensor(batch, "policy").to(device),
            values,
            value_weight,
        )
    return model.compute_loss_from_actions(
        boards,
        _batch_tensor(batch, "action").to(device),
        values,
        value_weight,
    )


def _compute_legal_masked_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor | list[str]],
    boards: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    value_weight: float,
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
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
        target_action = target_policy.argmax(dim=-1)
    else:
        target_action = _batch_tensor(batch, "action").to(device).long()
        policy_loss = F.cross_entropy(masked_logits, target_action)

    target_is_legal = legal_mask.gather(1, target_action.unsqueeze(1)).squeeze(1)
    if not bool(target_is_legal.all().item()):
        raise ValueError("legal_policy_loss received a target action that is illegal for its FEN")

    policy_acc = (masked_logits.argmax(dim=-1) == target_action).float().mean()
    value_loss = F.mse_loss(value, values)
    loss = policy_loss + value_weight * value_loss
    return loss, {
        "policy_loss": policy_loss.detach(),
        "policy_acc": policy_acc.detach(),
        "value_loss": value_loss.detach(),
    }


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
