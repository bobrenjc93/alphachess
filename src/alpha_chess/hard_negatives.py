"""Mine model top-wrong moves as bad-action replay data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.model import load_checkpoint
from alpha_chess.train import _legal_action_mask_from_fens, resolve_device


@dataclass
class HardNegativeConfig:
    checkpoint: str
    data: str | list[str]
    out: str
    batch_size: int = 256
    chunk_size: int = 1024
    bad_actions_per_position: int = 1
    prefer_action_labels: bool = True
    device: str = "auto"


def mine_hard_negatives(config: HardNegativeConfig) -> list[Path]:
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
    model.eval()

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _HardNegativeWriter(out_dir, max(1, int(config.chunk_size)))
    positions = 0
    hard_negatives = 0
    hard_negative_actions = 0

    with torch.no_grad():
        for batch in loader:
            if "fen" not in batch:
                raise ValueError("hard-negative mining requires replay data with FENs")
            fens = batch["fen"]
            if not isinstance(fens, list):
                raise TypeError("Batch FENs must be a list of strings")

            boards = batch["board"].to(device)
            targets = _target_actions(batch).to(device)
            policy_logits, _value = model(boards)
            legal_mask = _legal_action_mask_from_fens(fens, device)
            target_is_legal = legal_mask.gather(1, targets.unsqueeze(1)).squeeze(1)
            if not bool(target_is_legal.all().item()):
                raise ValueError("hard-negative mining received an illegal target action")

            masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
            bad_actions = _top_wrong_predictions(
                masked_logits,
                targets,
                max_bad_actions=config.bad_actions_per_position,
            )
            hard_negative_mask = bad_actions >= 0
            if hard_negative_mask.ndim > 1:
                hard_negative_positions = hard_negative_mask.any(dim=1)
            else:
                hard_negative_positions = hard_negative_mask
            hard_negatives += int(hard_negative_positions.sum().item())
            hard_negative_actions += int(hard_negative_mask.sum().item())
            positions += int(targets.numel())

            writer.add_batch(
                boards=batch["board"].cpu().numpy().astype(np.float32),
                actions=targets.cpu().numpy().astype(np.int64),
                values=batch["value"].cpu().numpy().astype(np.float32),
                fens=fens,
                bad_actions=bad_actions.cpu().numpy().astype(np.int64),
            )

    paths = writer.finish()
    summary = [
        f"checkpoint={config.checkpoint}",
        f"data={config.data}",
        f"positions={positions}",
        f"hard_negative_positions={hard_negatives}",
        f"hard_negative_actions={hard_negative_actions}",
        f"bad_actions_per_position={config.bad_actions_per_position}",
        f"top1_error_rate={hard_negatives / positions if positions else 0.0:.6f}",
        f"chunks={len(paths)}",
    ]
    (out_dir / "hard_negative_summary.txt").write_text("\n".join(summary) + "\n")
    return paths


def _target_actions(batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
    if "action" in batch:
        return batch["action"].long()
    if "policy" in batch:
        return batch["policy"].argmax(dim=-1).long()
    raise ValueError("hard-negative mining requires action labels or dense policies")


def _top_wrong_predictions(
    masked_logits: torch.Tensor,
    target_action: torch.Tensor,
    max_bad_actions: int = 1,
) -> torch.Tensor:
    max_bad_actions = max(1, int(max_bad_actions))
    top_actions = masked_logits.argmax(dim=-1)
    wrong_top = top_actions != target_action.long()

    if max_bad_actions == 1:
        bad_actions = torch.full_like(target_action.long(), -1)
        bad_actions[wrong_top] = top_actions[wrong_top]
        return bad_actions

    width = int(masked_logits.shape[1])
    candidate_count = min(width, max_bad_actions + 1)
    ranked_actions = torch.topk(masked_logits, k=candidate_count, dim=-1).indices
    bad_actions = torch.full(
        (target_action.shape[0], max_bad_actions),
        -1,
        dtype=torch.long,
        device=target_action.device,
    )
    for row in range(target_action.shape[0]):
        if not bool(wrong_top[row].item()):
            continue
        wrong_candidates = ranked_actions[row][ranked_actions[row] != target_action[row]]
        count = min(max_bad_actions, int(wrong_candidates.numel()))
        if count:
            bad_actions[row, :count] = wrong_candidates[:count]
    return bad_actions


class _HardNegativeWriter:
    def __init__(self, out_dir: Path, chunk_size: int) -> None:
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.paths: list[Path] = []
        self.boards: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.fens: list[str] = []
        self.bad_actions: list[np.ndarray] = []

    def add_batch(
        self,
        boards: np.ndarray,
        actions: np.ndarray,
        values: np.ndarray,
        fens: list[str],
        bad_actions: np.ndarray,
    ) -> None:
        start = 0
        while start < len(actions):
            remaining = self.chunk_size - self._size()
            stop = min(start + remaining, len(actions))
            self.boards.append(boards[start:stop])
            self.actions.append(actions[start:stop])
            self.values.append(values[start:stop])
            self.fens.extend(fens[start:stop])
            self.bad_actions.append(bad_actions[start:stop])
            start = stop
            if self._size() >= self.chunk_size:
                self._flush()

    def finish(self) -> list[Path]:
        if self._size() > 0:
            self._flush()
        return self.paths

    def _size(self) -> int:
        return sum(len(actions) for actions in self.actions)

    def _flush(self) -> None:
        index = len(self.paths)
        path = self.out_dir / f"hard_negatives_{index:06d}.npz"
        np.savez_compressed(
            path,
            boards=np.concatenate(self.boards, axis=0),
            actions=np.concatenate(self.actions, axis=0),
            values=np.concatenate(self.values, axis=0),
            fens=np.asarray(self.fens),
            bad_actions=np.concatenate(self.bad_actions, axis=0),
            source=np.asarray("hard_negatives"),
        )
        self.paths.append(path)
        self.boards.clear()
        self.actions.clear()
        self.values.clear()
        self.fens.clear()
        self.bad_actions.clear()
