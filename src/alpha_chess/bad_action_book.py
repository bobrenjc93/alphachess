"""Exact-position action filtering from mined replay data."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

ActionBook = dict[str, frozenset[int]]
BadActionBook = ActionBook
GoodActionBook = ActionBook


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


def load_good_action_book(
    paths: str | list[str] | tuple[str, ...] | None,
    *,
    policy_top_k: int = 1,
) -> GoodActionBook | None:
    if not paths:
        return None
    policy_top_k = max(1, int(policy_top_k))
    path_list = [paths] if isinstance(paths, str) else list(paths)
    mutable: dict[str, set[int]] = {}
    for path_text in path_list:
        path = Path(path_text)
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        for file_path in files:
            with np.load(file_path, allow_pickle=True) as data:
                if "fens" not in data:
                    raise ValueError(f"{file_path} must contain fens")
                if "actions" not in data and "policies" not in data:
                    raise ValueError(f"{file_path} must contain actions or policies")
                actions = data["actions"] if "actions" in data else None
                policies = data["policies"] if "policies" in data else None
                for row_index, fen in enumerate(data["fens"]):
                    row_actions: set[int] = set()
                    if actions is not None:
                        row_actions.add(int(actions[row_index]))
                    if policies is not None and (actions is None or policy_top_k > 1):
                        policy = np.asarray(policies[row_index]).reshape(-1)
                        if policy.size:
                            top_count = min(policy_top_k, int(policy.size))
                            top_indices = (
                                np.arange(policy.size)
                                if top_count >= policy.size
                                else np.argpartition(policy, -top_count)[-top_count:]
                            )
                            row_actions.update(
                                int(action)
                                for action in top_indices
                                if float(policy[int(action)]) > 0.0
                            )
                    row_actions = {action for action in row_actions if action >= 0}
                    if not row_actions:
                        continue
                    key = " ".join(str(fen).split()[:4])
                    mutable.setdefault(key, set()).update(row_actions)
    return {key: frozenset(actions) for key, actions in mutable.items()}


def filter_good_actions(
    board: chess.Board,
    actions: list[int],
    good_action_book: GoodActionBook | None,
) -> list[int]:
    if not good_action_book or not actions:
        return actions
    good_actions = good_action_book.get(position_key(board))
    if not good_actions:
        return actions
    filtered = [action for action in actions if action in good_actions]
    return filtered or actions


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
