"""PUCT Monte Carlo Tree Search for chess."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, action_to_move, legal_actions, terminal_value
from alpha_chess.evaluator import Evaluator


@dataclass
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    add_root_noise: bool = False


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)

    def expand(self, board: chess.Board, policy: np.ndarray, tactical_filter: bool = False) -> None:
        actions = legal_actions(board)
        if tactical_filter:
            actions = _filter_root_tactics(board, actions)
        if not actions:
            return

        priors = np.asarray(policy[actions], dtype=np.float64)
        total = float(priors.sum())
        if total <= 0 or not np.isfinite(total):
            priors = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        else:
            priors = priors / total

        self.children = {
            int(action): Node(prior=float(prior)) for action, prior in zip(actions, priors)
        }

    def add_exploration_noise(self, rng: np.random.Generator, alpha: float, fraction: float) -> None:
        if not self.children or alpha <= 0 or fraction <= 0:
            return
        actions = list(self.children)
        noise = rng.dirichlet([alpha] * len(actions))
        for action, sample in zip(actions, noise):
            child = self.children[action]
            child.prior = child.prior * (1.0 - fraction) + float(sample) * fraction


@dataclass
class SearchResult:
    root: Node
    visits: np.ndarray
    root_value: float

    def policy(self, temperature: float = 1.0) -> np.ndarray:
        counts = self.visits.astype(np.float64)
        policy = np.zeros_like(counts, dtype=np.float32)
        total_visits = float(counts.sum())
        if total_visits <= 0:
            return policy
        if temperature <= 0:
            policy[int(np.argmax(counts))] = 1.0
            return policy
        adjusted = np.power(counts, 1.0 / temperature)
        total = float(adjusted.sum())
        if total > 0:
            policy = (adjusted / total).astype(np.float32)
        return policy

    def select_action(self, temperature: float, rng: np.random.Generator) -> int | None:
        policy = self.policy(temperature)
        nonzero = np.flatnonzero(policy > 0)
        if len(nonzero) == 0:
            return None
        if temperature <= 0:
            return int(nonzero[np.argmax(policy[nonzero])])
        return int(rng.choice(np.arange(ACTION_SIZE), p=policy))


class AlphaZeroMCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()

    def run(self, board: chess.Board) -> SearchResult:
        root = Node(prior=1.0)
        value = terminal_value(board)
        if value is None:
            policy, value = self.evaluator(board)
            root.expand(board, policy, tactical_filter=True)
            if self.config.add_root_noise:
                root.add_exploration_noise(
                    self.rng, self.config.dirichlet_alpha, self.config.dirichlet_fraction
                )

        for _ in range(max(0, self.config.simulations)):
            scratch = board.copy(stack=False)
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self._select_child(node)
                move = action_to_move(action, scratch)
                if move is None:
                    break
                scratch.push(move)
                search_path.append(node)

            leaf_value = terminal_value(scratch)
            if leaf_value is None:
                policy, leaf_value = self.evaluator(scratch)
                node.expand(scratch, policy)

            self._backpropagate(search_path, float(leaf_value))

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return SearchResult(root=root, visits=visits, root_value=root.value)

    def _select_child(self, node: Node) -> tuple[int, Node]:
        _, action, child = max(
            (self._ucb_score(node, action, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def _ucb_score(self, parent: Node, _action: int, child: Node) -> float:
        prior_score = (
            self.config.c_puct
            * child.prior
            * np.sqrt(parent.visit_count + 1.0)
            / (child.visit_count + 1.0)
        )
        # Child value is from the child side-to-move perspective; negate it for parent.
        value_score = -child.value
        return float(value_score + prior_score)

    @staticmethod
    def _backpropagate(search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value


def _filter_root_tactics(board: chess.Board, actions: list[int]) -> list[int]:
    mate_actions: list[int] = []
    safe_actions: list[int] = []

    for action in actions:
        move = action_to_move(action, board)
        if move is None:
            continue
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            mate_actions.append(action)
        elif not _side_to_move_has_mate_in_one(child):
            safe_actions.append(action)

    if mate_actions:
        return mate_actions
    if safe_actions:
        return safe_actions
    return actions


def _side_to_move_has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            return True
    return False
