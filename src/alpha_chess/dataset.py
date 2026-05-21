"""Self-play dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_chess.chess_env import (
    ACTION_SIZE,
    color_mirror_action,
    color_mirror_board,
    color_mirror_policy,
    encode_board,
)

Sample = dict[str, torch.Tensor | str]


class SelfPlayDataset(Dataset):
    """Dataset backed by AlphaChess self-play NPZ files."""

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        in_memory: bool = False,
        color_mirror_augmentation: bool = False,
        prefer_action_labels: bool = False,
    ) -> None:
        if isinstance(data_dir, (str, Path)):
            data_dirs = [Path(data_dir)]
        else:
            data_dirs = [Path(path) for path in data_dir]

        self.color_mirror_augmentation = bool(color_mirror_augmentation)
        self.prefer_action_labels = bool(prefer_action_labels)
        self.files: list[Path] = []
        self.source_names = [str(path) for path in data_dirs]
        self.file_source_ids: list[int] = []
        for source_id, directory in enumerate(data_dirs):
            if directory.is_file() and directory.suffix == ".npz":
                self.files.append(directory)
                self.file_source_ids.append(source_id)
            else:
                files = sorted(directory.glob("*.npz"))
                self.files.extend(files)
                self.file_source_ids.extend([source_id] * len(files))

        if not self.files:
            raise FileNotFoundError(f"No .npz self-play files found in {data_dirs}")

        self.lengths: list[int] = []
        self.cache: dict[Path, dict[str, np.ndarray]] | None = {} if in_memory else None
        for path in self.files:
            data = np.load(path)
            if self.color_mirror_augmentation and "fens" not in data.files:
                raise ValueError(
                    f"color_mirror_augmentation requires stored FENs, but {path} has none"
                )
            self.lengths.append(int(data["boards"].shape[0]))
            if self.cache is not None:
                self.cache[path] = {
                    key: data[key]
                    for key in data.files
                    if key
                    in {
                        "boards",
                        "values",
                        "policies",
                        "actions",
                        "fens",
                        "bad_actions",
                        "bad_action_deltas",
                        "value_deltas",
                    }
                }

        self.cumsum = np.cumsum([0] + self.lengths)
        self.base_length = int(self.cumsum[-1])

    def __len__(self) -> int:
        if self.color_mirror_augmentation:
            return self.base_length * 2
        return self.base_length

    def source_sample_weights(
        self,
        source_weights: list[float],
        *,
        max_source_repeat: float | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Return per-sample weights that balance input data sources.

        Each source receives total probability mass proportional to its source
        weight, independent of how many positions it contains. When
        max_source_repeat is set, a source's probability mass is capped so its
        expected samples per position per epoch cannot exceed that value.
        """

        if len(source_weights) != len(self.source_names):
            raise ValueError(
                f"data_weights has {len(source_weights)} entries, "
                f"but dataset has {len(self.source_names)} data sources"
            )
        weights = np.asarray(source_weights, dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("data_weights must be finite non-negative values")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one data_weights entry must be positive")

        positions_per_source = np.zeros(len(self.source_names), dtype=np.float64)
        for source_id, length in zip(self.file_source_ids, self.lengths):
            positions_per_source[source_id] += length
        empty_weighted_sources = [
            self.source_names[source_id]
            for source_id, (positions, weight) in enumerate(zip(positions_per_source, weights))
            if positions <= 0 and weight > 0
        ]
        if empty_weighted_sources:
            raise ValueError(
                "data_weights assigns positive weight to empty data sources: "
                + ", ".join(empty_weighted_sources)
            )

        source_masses = weights / weights.sum()
        if max_source_repeat is not None:
            if not np.isfinite(max_source_repeat) or max_source_repeat <= 0:
                raise ValueError("max_source_repeat must be a finite positive value")
            epoch_samples = int(num_samples) if num_samples is not None else len(self)
            if epoch_samples <= 0:
                raise ValueError("num_samples must be positive when max_source_repeat is set")
            effective_positions = positions_per_source * (
                2 if self.color_mirror_augmentation else 1
            )
            source_masses = _cap_source_masses(
                source_masses,
                max_source_repeat * effective_positions / epoch_samples,
            )

        sample_weights = np.zeros(self.base_length, dtype=np.float64)
        for file_index, (source_id, length) in enumerate(zip(self.file_source_ids, self.lengths)):
            source_positions = positions_per_source[source_id]
            if source_positions <= 0:
                continue
            start = int(self.cumsum[file_index])
            end = int(self.cumsum[file_index + 1])
            sample_weights[start:end] = source_masses[source_id] / source_positions

        if self.color_mirror_augmentation:
            sample_weights = np.concatenate([sample_weights, sample_weights])
        return torch.as_tensor(sample_weights, dtype=torch.double)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        mirrored = False
        if self.color_mirror_augmentation and index >= self.base_length:
            index -= self.base_length
            mirrored = True
        file_index = int(np.searchsorted(self.cumsum[1:], index, side="right"))
        local_index = index - int(self.cumsum[file_index])
        path = self.files[file_index]
        data = self.cache[path] if self.cache is not None else np.load(path)

        if "policies" not in data and "actions" not in data:
            raise KeyError(f"{path} has neither 'policies' nor 'actions'")

        board = chess.Board(str(data["fens"][local_index])) if mirrored else None
        mirrored_board = color_mirror_board(board) if board is not None else None

        sample = {
            "board": (
                torch.from_numpy(encode_board(mirrored_board)).float()
                if mirrored_board is not None
                else torch.from_numpy(data["boards"][local_index]).float()
            ),
            "value": torch.tensor(float(data["values"][local_index]), dtype=torch.float32),
            "source_id": torch.tensor(self.file_source_ids[file_index], dtype=torch.long),
        }
        use_policy = "policies" in data and not (
            self.prefer_action_labels and "actions" in data
        )
        if use_policy:
            policy = data["policies"][local_index]
            if mirrored_board is not None and board is not None:
                policy = color_mirror_policy(policy, board)
            sample["policy"] = torch.from_numpy(policy).float()
        if "actions" in data:
            action = int(data["actions"][local_index])
            if mirrored_board is not None and board is not None:
                action = color_mirror_action(action, board)
            sample["action"] = torch.tensor(action, dtype=torch.long)
        if "fens" in data:
            sample["fen"] = (
                mirrored_board.fen()
                if mirrored_board is not None
                else str(data["fens"][local_index])
            )
        if "bad_actions" in data:
            bad_action_values = np.asarray(data["bad_actions"][local_index])
            if bad_action_values.ndim == 0:
                bad_actions = [int(bad_action_values)]
                scalar_bad_action = True
            else:
                bad_actions = [int(action) for action in bad_action_values.tolist()]
                scalar_bad_action = False
            if mirrored_board is not None and board is not None:
                bad_actions = [
                    color_mirror_action(action, board) if action >= 0 else action
                    for action in bad_actions
                ]
            sample["bad_action"] = torch.tensor(
                bad_actions[0] if scalar_bad_action else bad_actions,
                dtype=torch.long,
            )
            bad_action_delta_values = _bad_action_delta_values(data, local_index)
            if bad_action_delta_values is not None:
                sample["bad_action_delta"] = torch.from_numpy(
                    bad_action_delta_values.astype(np.float32, copy=False)
                )
        return sample

    def write_index(self, path: str | Path | None = None) -> Path:
        output = Path(path) if path is not None else self.files[0].parent / "index.json"
        output.write_text(
            json.dumps(
                {
                    "files": [
                        {"path": str(path), "positions": length}
                        for path, length in zip(self.files, self.lengths)
                    ],
                    "total_positions": len(self),
                },
                indent=2,
            )
        )
        return output


def _cap_source_masses(source_masses: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Project source masses onto per-source upper caps while preserving ratios."""

    caps = np.minimum(caps, 1.0)
    if np.any(caps < 0) or float(caps.sum()) <= 0:
        raise ValueError("max_source_repeat caps all data sources to zero")
    if float(caps.sum()) < 1.0:
        raise ValueError("max_source_repeat is too low to sample one full epoch")

    adjusted = np.zeros_like(source_masses)
    remaining = np.ones_like(source_masses, dtype=bool)
    remaining_mass = 1.0
    while bool(remaining.any()):
        desired = source_masses[remaining]
        desired_sum = float(desired.sum())
        if desired_sum <= 0:
            break
        scaled = desired / desired_sum * remaining_mass
        remaining_indices = np.flatnonzero(remaining)
        cap_values = caps[remaining]
        over_cap = scaled > cap_values
        if not bool(over_cap.any()):
            adjusted[remaining_indices] = scaled
            break
        capped_indices = remaining_indices[over_cap]
        adjusted[capped_indices] = cap_values[over_cap]
        remaining_mass -= float(cap_values[over_cap].sum())
        remaining[capped_indices] = False
        if remaining_mass <= 0:
            break

    total = float(adjusted.sum())
    if total <= 0:
        raise ValueError("max_source_repeat caps all data sources to zero")
    return adjusted / total


def collate_samples(samples: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    """Collate dense self-play policies or sparse expert action labels."""

    batch = {
        "board": torch.stack([sample["board"] for sample in samples]),
        "value": torch.stack([sample["value"] for sample in samples]),
        "source_id": torch.stack([sample["source_id"] for sample in samples]),
    }

    has_policy = any("policy" in sample for sample in samples)
    has_action = any("action" in sample for sample in samples)
    if has_policy:
        policies: list[torch.Tensor] = []
        for sample in samples:
            if "policy" in sample:
                policies.append(sample["policy"])
            elif "action" in sample:
                policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
                policy[int(sample["action"])] = 1.0
                policies.append(policy)
            else:
                raise KeyError("Sample has neither policy nor action")
        batch["policy"] = torch.stack(policies)
    elif has_action:
        batch["action"] = torch.stack([sample["action"] for sample in samples])
    else:
        raise KeyError("Batch has neither policies nor actions")

    if all("fen" in sample for sample in samples):
        batch["fen"] = [str(sample["fen"]) for sample in samples]

    if any("bad_action" in sample for sample in samples):
        bad_actions: list[torch.Tensor] = []
        bad_action_deltas: list[torch.Tensor] = []
        max_bad_actions = 1
        for sample in samples:
            if "bad_action" in sample:
                bad_action = sample["bad_action"]
                if not isinstance(bad_action, torch.Tensor):
                    raise TypeError("bad_action sample field must be a tensor")
                bad_action = bad_action.long().reshape(-1)
            else:
                bad_action = torch.tensor([-1], dtype=torch.long)
            if "bad_action_delta" in sample:
                bad_action_delta = sample["bad_action_delta"]
                if not isinstance(bad_action_delta, torch.Tensor):
                    raise TypeError("bad_action_delta sample field must be a tensor")
                bad_action_delta = bad_action_delta.float().reshape(-1)
                if bad_action_delta.numel() < bad_action.numel():
                    bad_action_delta = torch.cat(
                        [
                            bad_action_delta,
                            torch.zeros(
                                bad_action.numel() - bad_action_delta.numel(),
                                dtype=torch.float32,
                            ),
                        ]
                    )
                elif bad_action_delta.numel() > bad_action.numel():
                    bad_action_delta = bad_action_delta[: bad_action.numel()]
            else:
                bad_action_delta = torch.zeros(bad_action.numel(), dtype=torch.float32)
            bad_actions.append(bad_action)
            bad_action_deltas.append(bad_action_delta)
            max_bad_actions = max(max_bad_actions, int(bad_action.numel()))

        padded_bad_actions = torch.full(
            (len(samples), max_bad_actions),
            -1,
            dtype=torch.long,
        )
        padded_bad_action_deltas = torch.zeros(
            (len(samples), max_bad_actions),
            dtype=torch.float32,
        )
        for row, (bad_action, bad_action_delta) in enumerate(
            zip(bad_actions, bad_action_deltas)
        ):
            padded_bad_actions[row, : bad_action.numel()] = bad_action
            padded_bad_action_deltas[row, : bad_action_delta.numel()] = bad_action_delta
        batch["bad_action"] = (
            padded_bad_actions.squeeze(1)
            if max_bad_actions == 1
            else padded_bad_actions
        )
        batch["bad_action_delta"] = (
            padded_bad_action_deltas.squeeze(1)
            if max_bad_actions == 1
            else padded_bad_action_deltas
        )

    return batch


def _bad_action_delta_values(
    data: dict[str, np.ndarray],
    local_index: int,
) -> np.ndarray | None:
    if "bad_action_deltas" in data:
        return np.asarray(data["bad_action_deltas"][local_index], dtype=np.float32)
    if "value_deltas" not in data:
        return None
    value_delta = np.asarray(data["value_deltas"][local_index], dtype=np.float32)
    if value_delta.ndim == 0:
        return None
    return value_delta
