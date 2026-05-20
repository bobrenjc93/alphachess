"""Exact-position bad-action filtering from mined replay data."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

BadActionBook = dict[str, frozenset[int]]


def position_key(board: chess.Board) -> str:
    """Return a FEN key without move counters."""

    return " ".join(board.fen().split()[:4])


def load_bad_action_book(paths: str | list[str] | tuple[str, ...] | None) -> BadActionBook | None:
    if not paths:
        return None
    path_list = [paths] if isinstance(paths, str) else list(paths)
    mutable: dict[str, set[int]] = {}
    for path_text in path_list:
        path = Path(path_text)
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        for file_path in files:
            with np.load(file_path, allow_pickle=True) as data:
                if "fens" not in data or "bad_actions" not in data:
                    raise ValueError(f"{file_path} must contain fens and bad_actions arrays")
                bad_actions = data["bad_actions"]
                if bad_actions.ndim == 1:
                    bad_actions = bad_actions[:, None]
                for fen, row in zip(data["fens"], bad_actions):
                    actions = {int(action) for action in np.asarray(row).reshape(-1) if int(action) >= 0}
                    if not actions:
                        continue
                    key = " ".join(str(fen).split()[:4])
                    mutable.setdefault(key, set()).update(actions)
    return {key: frozenset(actions) for key, actions in mutable.items()}


def filter_bad_actions(
    board: chess.Board,
    actions: list[int],
    bad_action_book: BadActionBook | None,
) -> list[int]:
    if not bad_action_book or not actions:
        return actions
    bad_actions = bad_action_book.get(position_key(board))
    if not bad_actions:
        return actions
    filtered = [action for action in actions if action not in bad_actions]
    return filtered or actions
