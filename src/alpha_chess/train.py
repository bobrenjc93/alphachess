"""Training loop for AlphaChess checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

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
            loss, parts = _compute_batch_loss(model, batch, device, config.value_weight)
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
            latest_metrics.update(_evaluate_loss(model, val_loader, device, config.value_weight))

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
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    policy_losses: list[float] = []
    policy_accs: list[float] = []
    value_losses: list[float] = []
    for batch in loader:
        loss, parts = _compute_batch_loss(model, batch, device, value_weight)
        losses.append(float(loss.item()))
        policy_losses.append(float(parts["policy_loss"].item()))
        policy_accs.append(float(parts["policy_acc"].item()))
        value_losses.append(float(parts["value_loss"].item()))
    return {
        "val_loss": sum(losses) / max(1, len(losses)),
        "val_policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "val_policy_acc": sum(policy_accs) / max(1, len(policy_accs)),
        "val_value_loss": sum(value_losses) / max(1, len(value_losses)),
    }


def _compute_batch_loss(
    model: ChessNet,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    value_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boards = batch["board"].to(device)
    values = batch["value"].to(device)
    if "policy" in batch:
        return model.compute_loss(boards, batch["policy"].to(device), values, value_weight)
    return model.compute_loss_from_actions(boards, batch["action"].to(device), values, value_weight)
