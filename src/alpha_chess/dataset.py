"""Self-play dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_chess.chess_env import ACTION_SIZE

Sample = dict[str, torch.Tensor | str]


class SelfPlayDataset(Dataset):
    """Dataset backed by AlphaChess self-play NPZ files."""

    def __init__(self, data_dir: str | Path | list[str | Path], in_memory: bool = False) -> None:
        if isinstance(data_dir, (str, Path)):
            data_dirs = [Path(data_dir)]
        else:
            data_dirs = [Path(path) for path in data_dir]

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
            self.lengths.append(int(data["boards"].shape[0]))
            if self.cache is not None:
                self.cache[path] = {
                    key: data[key]
                    for key in data.files
                    if key in {"boards", "values", "policies", "actions", "fens"}
                }

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def source_sample_weights(self, source_weights: list[float]) -> torch.Tensor:
        """Return per-sample weights that balance input data sources.

        Each source receives total probability mass proportional to its source
        weight, independent of how many positions it contains.
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

        sample_weights = np.zeros(len(self), dtype=np.float64)
        for file_index, (source_id, length) in enumerate(zip(self.file_source_ids, self.lengths)):
            source_positions = positions_per_source[source_id]
            if source_positions <= 0:
                continue
            start = int(self.cumsum[file_index])
            end = int(self.cumsum[file_index + 1])
            sample_weights[start:end] = weights[source_id] / source_positions

        return torch.as_tensor(sample_weights, dtype=torch.double)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.cumsum[1:], index, side="right"))
        local_index = index - int(self.cumsum[file_index])
        path = self.files[file_index]
        data = self.cache[path] if self.cache is not None else np.load(path)

        if "policies" not in data and "actions" not in data:
            raise KeyError(f"{path} has neither 'policies' nor 'actions'")

        sample = {
            "board": torch.from_numpy(data["boards"][local_index]).float(),
            "value": torch.tensor(float(data["values"][local_index]), dtype=torch.float32),
            "source_id": torch.tensor(self.file_source_ids[file_index], dtype=torch.long),
        }
        if "policies" in data:
            sample["policy"] = torch.from_numpy(data["policies"][local_index]).float()
        if "actions" in data:
            sample["action"] = torch.tensor(int(data["actions"][local_index]), dtype=torch.long)
        if "fens" in data:
            sample["fen"] = str(data["fens"][local_index])
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

    return batch
