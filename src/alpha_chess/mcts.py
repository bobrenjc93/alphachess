"""PUCT Monte Carlo Tree Search for chess."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, action_to_move, legal_actions, terminal_value
from alpha_chess.bad_action_book import BadActionBook, filter_bad_actions
from alpha_chess.evaluator import PIECE_VALUES, Evaluator

MATE_SCORE_CP = 100_000
TACTICAL_MATERIAL_CANDIDATE_LIMIT = 16
QUIET_MATERIAL_CANDIDATE_LIMIT = 8
BOARD_SIZE = 8
KING_SHELTER_NEAR_PAWN_CP = 80
KING_SHELTER_FAR_PAWN_CP = 25
KING_SHELTER_OPEN_FILE_CP = 70
KING_SHELTER_SLIDER_FILE_CP = 45
KING_SHELTER_MIN_NON_KING_MATERIAL_CP = 2400


@dataclass
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    policy_prior_temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    add_root_noise: bool = False
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250
    root_king_safety_search_plies: int = 0
    root_king_safety_max_loss_cp: int = 250
    leaf_material_value_weight: float = 0.0
    leaf_material_search_plies: int = 0
    root_bad_action_book: BadActionBook | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.policy_prior_temperature)
            or self.policy_prior_temperature <= 0
        ):
            raise ValueError("policy_prior_temperature must be finite and positive")
        if (
            not np.isfinite(self.leaf_material_value_weight)
            or not 0.0 <= self.leaf_material_value_weight <= 1.0
        ):
            raise ValueError("leaf_material_value_weight must be between 0 and 1")
        self.leaf_material_search_plies = max(0, int(self.leaf_material_search_plies))


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
        policy_prior_temperature: float = 1.0,
        tactical_filter_plies: int = 0,
        material_filter_plies: int = 0,
        material_max_loss_cp: int = 250,
        king_safety_filter_plies: int = 0,
        king_safety_max_loss_cp: int = 250,
        bad_action_book: BadActionBook | None = None,
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
        if king_safety_filter_plies > 0 and not forced_mate_found:
            actions = _filter_root_king_safety(
                board,
                actions,
                king_safety_filter_plies,
                king_safety_max_loss_cp,
            )
        actions = filter_bad_actions(board, actions, bad_action_book)
        if not actions:
            return

        priors = np.asarray(policy[actions], dtype=np.float64)
        priors = np.maximum(priors, 0.0)
        if policy_prior_temperature != 1.0:
            priors = np.power(priors, 1.0 / policy_prior_temperature)
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
            if temperature <= 0:
                action = self._deterministic_action()
                if action is not None:
                    policy[action] = 1.0
            return policy
        if temperature <= 0:
            action = self._deterministic_action()
            if action is not None:
                policy[action] = 1.0
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
            return self._deterministic_action()
        return int(rng.choice(np.arange(ACTION_SIZE), p=policy))

    def child_for_action(self, action: int) -> Node | None:
        return self.root.children.get(action)

    def _deterministic_action(self) -> int | None:
        if not self.root.children:
            return None
        max_visits = max(
            (int(self.visits[action]) for action in self.root.children),
            default=0,
        )
        candidates = [
            action
            for action in self.root.children
            if int(self.visits[action]) == max_visits
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda action: (self.root.children[action].prior, -action))


def advance_root(root: Node | None, action: int) -> Node | None:
    """Return the subtree reached by ``action`` if it is already available."""

    if root is None:
        return None
    return root.children.get(action)


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

    def run(self, board: chess.Board, root: Node | None = None) -> SearchResult:
        root = root or Node(prior=1.0)
        value = terminal_value(board)
        if value is None:
            if not root.expanded():
                policy, value = self.evaluator(board)
                root.expand(
                    board,
                    policy,
                    policy_prior_temperature=self.config.policy_prior_temperature,
                    tactical_filter_plies=self.config.root_mate_search_plies,
                    material_filter_plies=self.config.root_material_search_plies,
                    material_max_loss_cp=self.config.root_material_max_loss_cp,
                    king_safety_filter_plies=self.config.root_king_safety_search_plies,
                    king_safety_max_loss_cp=self.config.root_king_safety_max_loss_cp,
                    bad_action_book=self.config.root_bad_action_book,
                )
            else:
                self._filter_root_children(board, root)
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
                leaf_value = self._blend_leaf_value(scratch, float(leaf_value))
                node.expand(
                    scratch,
                    policy,
                    policy_prior_temperature=self.config.policy_prior_temperature,
                )

            self._backpropagate(search_path, float(leaf_value))

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return SearchResult(root=root, visits=visits, root_value=root.value)

    def _filter_root_children(self, board: chess.Board, root: Node) -> None:
        actions = list(root.children)
        if not actions:
            return

        forced_mate_found = False
        if self.config.root_mate_search_plies > 0:
            actions, forced_mate_found = _filter_root_tactics(
                board,
                actions,
                self.config.root_mate_search_plies,
            )
        if self.config.root_material_search_plies > 0 and not forced_mate_found:
            actions = _filter_root_material(
                board,
                actions,
                self.config.root_material_search_plies,
                self.config.root_material_max_loss_cp,
            )
        if self.config.root_king_safety_search_plies > 0 and not forced_mate_found:
            actions = _filter_root_king_safety(
                board,
                actions,
                self.config.root_king_safety_search_plies,
                self.config.root_king_safety_max_loss_cp,
            )
        actions = filter_bad_actions(board, actions, self.config.root_bad_action_book)
        root.children = {
            action: root.children[action] for action in actions if action in root.children
        }

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

    def _blend_leaf_value(self, board: chess.Board, neural_value: float) -> float:
        weight = self.config.leaf_material_value_weight
        if weight <= 0:
            return neural_value
        material = _material_search_value(
            board,
            self.config.leaf_material_search_plies,
        )
        return float((1.0 - weight) * neural_value + weight * material)

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
    if material_search_plies <= 0:
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
        score = _material_worst_depth_score(
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


def _filter_root_king_safety(
    board: chess.Board,
    actions: list[int],
    search_plies: int,
    max_loss_cp: int,
) -> list[int]:
    search_plies = max(0, search_plies)
    max_loss_cp = max(0, max_loss_cp)
    if search_plies <= 0:
        return actions

    perspective = board.turn
    baseline = _static_safety_score_cp(board, perspective)
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
        immediate_score = _static_safety_score_cp(child, perspective)
        search_score = _static_safety_full_width_score(
            child,
            search_plies,
            perspective,
            full_width_cache,
            quiescence_cache,
        )
        score = min(immediate_score, search_score)
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


def _material_search_value(board: chess.Board, search_plies: int) -> float:
    perspective = board.turn
    if search_plies <= 0:
        score = _material_score_cp(board, perspective)
    else:
        score = _material_full_width_score(
            board,
            search_plies,
            perspective,
            {},
            {},
        )
    return float(np.tanh(score / 1200.0))


def _material_worst_depth_score(
    board: chess.Board,
    max_plies: int,
    perspective: chess.Color,
    cache: dict[tuple[str, int, bool], int],
    quiescence_cache: dict[tuple[str, int, bool], int],
) -> int:
    max_plies = max(1, max_plies)
    scores = [
        _material_full_width_score(
            board,
            plies,
            perspective,
            cache,
            quiescence_cache,
        )
        for plies in range(1, max_plies + 1)
    ]
    return min(scores)


def _static_safety_full_width_score(
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
        material = _material_quiescence_score(board, 1, perspective, quiescence_cache)
        score = material + _king_safety_score_cp(board, perspective)
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
                _static_safety_full_width_score(
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
                _static_safety_full_width_score(
                    child,
                    plies - 1,
                    perspective,
                    cache,
                    quiescence_cache,
                ),
            )

    cache[key] = best
    return best


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
    return moves[:TACTICAL_MATERIAL_CANDIDATE_LIMIT]


def _mate_forcing_moves(board: chess.Board) -> list[chess.Move]:
    checks: list[chess.Move] = []
    captures: list[chess.Move] = []
    for move in board.legal_moves:
        if board.gives_check(move):
            checks.append(move)
        elif board.is_capture(move) or move.promotion:
            captures.append(move)

    checks.sort(key=lambda move: _move_material_priority(board, move), reverse=True)
    captures.sort(key=lambda move: _move_material_priority(board, move), reverse=True)
    slots = max(0, TACTICAL_MATERIAL_CANDIDATE_LIMIT - len(checks))
    return checks + captures[:slots]


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


def _static_safety_score_cp(board: chess.Board, perspective: chess.Color) -> int:
    return _material_score_cp(board, perspective) + _king_safety_score_cp(board, perspective)


def _king_safety_score_cp(board: chess.Board, perspective: chess.Color) -> int:
    return _king_danger_cp(board, not perspective) - _king_danger_cp(board, perspective)


def _king_danger_cp(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return MATE_SCORE_CP

    danger = 0
    if board.turn == color and board.is_check():
        danger += 500

    enemy = not color
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    ring: list[chess.Square] = [king_square]
    for file_delta in (-1, 0, 1):
        for rank_delta in (-1, 0, 1):
            if file_delta == 0 and rank_delta == 0:
                continue
            square = _square_from_file_rank(king_file + file_delta, king_rank + rank_delta)
            if square is not None:
                ring.append(square)

    for square in ring:
        attackers = board.attackers(enemy, square)
        for attacker_square in attackers:
            attacker = board.piece_at(attacker_square)
            if attacker is None:
                continue
            danger += _king_attack_weight_cp(attacker.piece_type)

    danger += _king_shelter_danger_cp(board, color, king_square)
    return danger


def _king_shelter_danger_cp(
    board: chess.Board,
    color: chess.Color,
    king_square: chess.Square,
) -> int:
    if _non_king_material_score_cp(board) < KING_SHELTER_MIN_NON_KING_MATERIAL_CP:
        return 0

    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    home_rank = 0 if color == chess.WHITE else 7
    if abs(king_rank - home_rank) > 1:
        return 0

    forward = 1 if color == chess.WHITE else -1
    enemy = not color
    danger = 0
    for file_delta in (-1, 0, 1):
        file_idx = king_file + file_delta
        if not 0 <= file_idx < BOARD_SIZE:
            continue

        near_square = _square_from_file_rank(file_idx, king_rank + forward)
        far_square = _square_from_file_rank(file_idx, king_rank + 2 * forward)
        near_has_pawn = _has_friendly_pawn(board, near_square, color)
        far_has_pawn = _has_friendly_pawn(board, far_square, color)

        if not near_has_pawn:
            danger += KING_SHELTER_NEAR_PAWN_CP
            if not far_has_pawn:
                danger += KING_SHELTER_FAR_PAWN_CP

        if not _has_friendly_pawn_ahead(board, file_idx, king_rank, forward, color):
            danger += KING_SHELTER_OPEN_FILE_CP
            if _has_enemy_slider_on_file(board, file_idx, enemy):
                danger += KING_SHELTER_SLIDER_FILE_CP

    return danger


def _has_friendly_pawn(
    board: chess.Board,
    square: chess.Square | None,
    color: chess.Color,
) -> bool:
    if square is None:
        return False
    piece = board.piece_at(square)
    return piece is not None and piece.color == color and piece.piece_type == chess.PAWN


def _has_friendly_pawn_ahead(
    board: chess.Board,
    file_idx: int,
    king_rank: int,
    forward: int,
    color: chess.Color,
) -> bool:
    rank = king_rank + forward
    while 0 <= rank < BOARD_SIZE:
        square = chess.square(file_idx, rank)
        if _has_friendly_pawn(board, square, color):
            return True
        rank += forward
    return False


def _has_enemy_slider_on_file(
    board: chess.Board,
    file_idx: int,
    enemy: chess.Color,
) -> bool:
    for rank in range(BOARD_SIZE):
        piece = board.piece_at(chess.square(file_idx, rank))
        if (
            piece is not None
            and piece.color == enemy
            and piece.piece_type in {chess.ROOK, chess.QUEEN}
        ):
            return True
    return False


def _non_king_material_score_cp(board: chess.Board) -> int:
    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.KING:
            total += PIECE_VALUES.get(piece.piece_type, 0)
    return total


def _square_from_file_rank(file_idx: int, rank_idx: int) -> chess.Square | None:
    if 0 <= file_idx < BOARD_SIZE and 0 <= rank_idx < BOARD_SIZE:
        return chess.square(file_idx, rank_idx)
    return None


def _king_attack_weight_cp(piece_type: chess.PieceType) -> int:
    if piece_type == chess.PAWN:
        return 25
    if piece_type in {chess.KNIGHT, chess.BISHOP}:
        return 40
    if piece_type == chess.ROOK:
        return 55
    if piece_type == chess.QUEEN:
        return 85
    if piece_type == chess.KING:
        return 15
    return 0


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

    for move in _mate_forcing_moves(board):
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            cache[key] = True
            return True
        if plies > 1 and _all_replies_allow_mate(child, plies - 1, cache):
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
