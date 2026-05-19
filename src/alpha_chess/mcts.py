"""PUCT Monte Carlo Tree Search for chess."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, action_to_move, legal_actions, terminal_value
from alpha_chess.evaluator import PIECE_VALUES, Evaluator

MATE_SCORE_CP = 100_000
QUIET_MATERIAL_CANDIDATE_LIMIT = 8


@dataclass
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    add_root_noise: bool = False
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250


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

    def expand(
        self,
        board: chess.Board,
        policy: np.ndarray,
        tactical_filter_plies: int = 0,
        material_filter_plies: int = 0,
        material_max_loss_cp: int = 250,
    ) -> None:
        actions = legal_actions(board)
        forced_mate_found = False
        if tactical_filter_plies > 0:
            actions, forced_mate_found = _filter_root_tactics(
                board, actions, tactical_filter_plies
            )
        if material_filter_plies > 0 and not forced_mate_found:
            actions = _filter_root_material(
                board,
                actions,
                material_filter_plies,
                material_max_loss_cp,
            )
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
            root.expand(
                board,
                policy,
                tactical_filter_plies=self.config.root_mate_search_plies,
                material_filter_plies=self.config.root_material_search_plies,
                material_max_loss_cp=self.config.root_material_max_loss_cp,
            )
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


def _filter_root_tactics(
    board: chess.Board, actions: list[int], mate_search_plies: int
) -> tuple[list[int], bool]:
    mate_actions: list[int] = []
    safe_actions: list[int] = []
    mate_cache: dict[tuple[str, int], bool] = {}
    mate_search_plies = max(1, mate_search_plies)

    for action in actions:
        move = action_to_move(action, board)
        if move is None:
            continue
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate() or (
            mate_search_plies > 1
            and _all_replies_allow_mate(child, mate_search_plies - 1, mate_cache)
        ):
            mate_actions.append(action)
        elif not _side_to_move_can_force_mate(child, mate_search_plies, mate_cache):
            safe_actions.append(action)

    if mate_actions:
        return mate_actions, True
    if safe_actions:
        return safe_actions, False
    return actions, False


def _filter_root_material(
    board: chess.Board,
    actions: list[int],
    material_search_plies: int,
    max_loss_cp: int,
) -> list[int]:
    material_search_plies = max(0, material_search_plies)
    max_loss_cp = max(0, max_loss_cp)
    if material_search_plies <= 0 or max_loss_cp <= 0:
        return actions

    perspective = board.turn
    baseline = _material_score_cp(board, perspective)
    min_allowed = baseline - max_loss_cp
    full_width_cache: dict[tuple[str, int, bool], int] = {}
    quiescence_cache: dict[tuple[str, int, bool], int] = {}
    safe_actions: list[int] = []
    scored_actions: list[tuple[int, int]] = []

    for action in actions:
        move = action_to_move(action, board)
        if move is None:
            continue
        child = board.copy(stack=False)
        child.push(move)
        score = _material_full_width_score(
            child,
            material_search_plies,
            perspective,
            full_width_cache,
            quiescence_cache,
        )
        scored_actions.append((action, score))
        if score >= min_allowed:
            safe_actions.append(action)

    if safe_actions:
        return safe_actions
    if not scored_actions:
        return actions

    best_score = max(score for _, score in scored_actions)
    fallback_actions = [action for action, score in scored_actions if score == best_score]
    return fallback_actions if fallback_actions else [scored_actions[0][0]]


def _material_full_width_score(
    board: chess.Board,
    plies: int,
    perspective: chess.Color,
    cache: dict[tuple[str, int, bool], int],
    quiescence_cache: dict[tuple[str, int, bool], int],
) -> int:
    key = (board.fen(), plies, perspective)
    if key in cache:
        return cache[key]

    if plies <= 0 or board.is_game_over(claim_draw=True):
        score = _material_quiescence_score(board, 1, perspective, quiescence_cache)
        cache[key] = score
        return score

    moves = _material_candidate_moves(board)

    if board.turn == perspective:
        best = -MATE_SCORE_CP
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = max(
                best,
                _material_full_width_score(
                    child,
                    plies - 1,
                    perspective,
                    cache,
                    quiescence_cache,
                ),
            )
    else:
        best = MATE_SCORE_CP
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = min(
                best,
                _material_full_width_score(
                    child,
                    plies - 1,
                    perspective,
                    cache,
                    quiescence_cache,
                ),
            )

    cache[key] = best
    return best


def _material_candidate_moves(board: chess.Board) -> list[chess.Move]:
    tactical_moves = _material_tactical_moves(board)
    if tactical_moves:
        return tactical_moves

    quiet_moves = list(board.legal_moves)
    quiet_moves.sort(key=lambda move: _quiet_threat_priority(board, move), reverse=True)
    return quiet_moves[:QUIET_MATERIAL_CANDIDATE_LIMIT]


def _material_quiescence_score(
    board: chess.Board,
    plies: int,
    perspective: chess.Color,
    cache: dict[tuple[str, int, bool], int],
) -> int:
    key = (board.fen(), plies, perspective)
    if key in cache:
        return cache[key]

    stand_pat = _material_score_cp(board, perspective)
    if plies <= 0 or board.is_game_over(claim_draw=True):
        cache[key] = stand_pat
        return stand_pat

    moves = _material_tactical_moves(board)
    if not moves:
        cache[key] = stand_pat
        return stand_pat

    if board.turn == perspective:
        best = stand_pat
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = max(best, _material_quiescence_score(child, plies - 1, perspective, cache))
    else:
        best = stand_pat
        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            best = min(best, _material_quiescence_score(child, plies - 1, perspective, cache))

    cache[key] = best
    return best


def _material_tactical_moves(board: chess.Board) -> list[chess.Move]:
    moves = [
        move
        for move in board.legal_moves
        if board.is_capture(move) or move.promotion or board.gives_check(move)
    ]
    moves.sort(key=lambda move: _move_material_priority(board, move), reverse=True)
    return moves


def _move_material_priority(board: chess.Board, move: chess.Move) -> int:
    captured_value = _captured_piece_value(board, move)
    promotion_value = PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0
    check_bonus = 10_000 if board.gives_check(move) else 0
    moving_piece = board.piece_at(move.from_square)
    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
    return check_bonus + captured_value + promotion_value - moving_value


def _quiet_threat_priority(board: chess.Board, move: chess.Move) -> int:
    child = board.copy(stack=False)
    child.push(move)
    mover = not child.turn
    threat_score = 0
    for square, piece in child.piece_map().items():
        if piece.color != mover and child.is_attacked_by(mover, square):
            threat_score = max(threat_score, PIECE_VALUES.get(piece.piece_type, 0))

    moving_piece = board.piece_at(move.from_square)
    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
    return threat_score - moving_value


def _captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return PIECE_VALUES.get(captured.piece_type, 0) if captured else 0


def _material_score_cp(board: chess.Board, perspective: chess.Color) -> int:
    if board.is_checkmate():
        return -MATE_SCORE_CP if board.turn == perspective else MATE_SCORE_CP
    if board.is_game_over(claim_draw=True):
        return 0

    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        score += value if piece.color == perspective else -value
    return score


def _side_to_move_can_force_mate(
    board: chess.Board,
    plies: int,
    cache: dict[tuple[str, int], bool],
) -> bool:
    if plies <= 0 or board.is_game_over(claim_draw=True):
        return False
    key = (board.fen(), plies)
    if key in cache:
        return cache[key]

    for move in board.legal_moves:
        gives_check = board.gives_check(move)
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            cache[key] = True
            return True
        if plies > 1 and gives_check and _all_replies_allow_mate(child, plies - 1, cache):
            cache[key] = True
            return True

    cache[key] = False
    return False


def _all_replies_allow_mate(
    board: chess.Board,
    plies: int,
    cache: dict[tuple[str, int], bool],
) -> bool:
    if plies <= 0 or board.is_game_over(claim_draw=True):
        return False

    has_reply = False
    for reply in board.legal_moves:
        has_reply = True
        child = board.copy(stack=False)
        child.push(reply)
        if child.is_checkmate() or not _side_to_move_can_force_mate(child, plies - 1, cache):
            return False
    return has_reply
