"""Self-play dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SelfPlayDataset(Dataset):
    """Dataset backed by AlphaChess self-play NPZ files."""

    def __init__(self, data_dir: str | Path | list[str | Path], in_memory: bool = False) -> None:
        if isinstance(data_dir, (str, Path)):
            data_dirs = [Path(data_dir)]
        else:
            data_dirs = [Path(path) for path in data_dir]

        self.files: list[Path] = []
        for directory in data_dirs:
            if directory.is_file() and directory.suffix == ".npz":
                self.files.append(directory)
            else:
                self.files.extend(sorted(directory.glob("*.npz")))

        if not self.files:
            raise FileNotFoundError(f"No .npz self-play files found in {data_dirs}")

        self.lengths: list[int] = []
        self.cache: dict[Path, dict[str, np.ndarray]] | None = {} if in_memory else None
        for path in self.files:
            data = np.load(path)
            self.lengths.append(int(data["boards"].shape[0]))
            if self.cache is not None:
                self.cache[path] = {key: data[key] for key in data.files}

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self.cumsum[1:], index, side="right"))
        local_index = index - int(self.cumsum[file_index])
        path = self.files[file_index]
        data = self.cache[path] if self.cache is not None else np.load(path)

        return {
            "board": torch.from_numpy(data["boards"][local_index]).float(),
            "policy": torch.from_numpy(data["policies"][local_index]).float(),
            "value": torch.tensor(float(data["values"][local_index]), dtype=torch.float32),
        }

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
