import chess
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, action_to_move, legal_actions, move_to_action
from alpha_chess.bad_action_book import (
    load_bad_action_book,
    load_good_action_book,
    position_key,
)
from alpha_chess.evaluator import UniformEvaluator
from alpha_chess.mcts import (
    TACTICAL_MATERIAL_CANDIDATE_LIMIT,
    AlphaZeroMCTS,
    MCTSConfig,
    Node,
    SearchResult,
    _material_tactical_moves,
    _side_to_move_can_force_mate,
    _static_safety_candidate_moves,
    advance_root,
)


class SparsePolicyEvaluator:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def __call__(self, board: chess.Board) -> tuple[np.ndarray, float]:
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move_uci, weight in self.weights.items():
            action = move_to_action(chess.Move.from_uci(move_uci), board)
            policy[action] = weight
        return policy, 0.0


def test_mcts_returns_legal_policy() -> None:
    board = chess.Board()
    search = AlphaZeroMCTS(UniformEvaluator(), MCTSConfig(simulations=8))
    result = search.run(board)
    policy = result.policy(temperature=1.0)
    assert policy.sum() > 0
    action = result.select_action(temperature=0.0, rng=search.rng)
    assert action is not None
    move = action_to_move(action, board)
    assert move in board.legal_moves


def test_mcts_handles_terminal_board() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_checkmate()
    search = AlphaZeroMCTS(UniformEvaluator(), MCTSConfig(simulations=4))
    result = search.run(board)
    assert result.visits.sum() == 0


def test_mcts_can_reuse_advanced_root() -> None:
    board = chess.Board()
    search = AlphaZeroMCTS(UniformEvaluator(), MCTSConfig(simulations=4))
    result = search.run(board)
    action = result.select_action(temperature=0.0, rng=search.rng)
    assert action is not None
    reused_root = advance_root(result.root, action)
    assert reused_root is not None

    move = action_to_move(action, board)
    assert move is not None
    board.push(move)

    reused_result = search.run(board, root=reused_root)
    assert reused_result.root is reused_root
    assert reused_result.visits.sum() > 0


def test_mcts_applies_root_filters_to_reused_root() -> None:
    board = chess.Board("3r2k1/8/8/8/8/8/3N4/3Q2K1 w - - 0 1")
    blunder_action = move_to_action(chess.Move.from_uci("d2f3"), board)
    root = Node(prior=1.0)
    actions = legal_actions(board)
    root.children = {action: Node(prior=1.0 / len(actions)) for action in actions}

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=1,
            root_material_max_loss_cp=350,
        ),
    )
    result = search.run(board, root=root)

    assert result.root is root
    assert blunder_action not in result.root.children


def test_mcts_filters_exact_position_bad_action() -> None:
    board = chess.Board()
    bad_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    good_action = move_to_action(chess.Move.from_uci("d2d4"), board)
    evaluator = SparsePolicyEvaluator({"e2e4": 0.9, "d2d4": 0.1})

    search = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_bad_action_book={position_key(board): frozenset({bad_action})},
        ),
    )
    result = search.run(board)

    assert bad_action not in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) == good_action


def test_mcts_filters_exact_position_good_action_before_root_guards() -> None:
    board = chess.Board()
    teacher_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    alternative_action = move_to_action(chess.Move.from_uci("g1f3"), board)
    evaluator = SparsePolicyEvaluator({"g1f3": 0.9, "e2e4": 0.1})

    search = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_king_safety_search_plies=1,
            root_king_safety_max_loss_cp=100,
            root_good_action_book={position_key(board): frozenset({teacher_action})},
        ),
    )
    result = search.run(board)

    assert set(result.root.children) == {teacher_action}
    assert alternative_action not in result.root.children


def test_bad_action_book_ignores_move_counters(tmp_path) -> None:
    board = chess.Board()
    bad_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    path = tmp_path / "book.npz"
    np.savez_compressed(
        path,
        fens=np.asarray([board.fen()]),
        bad_actions=np.asarray([[bad_action]], dtype=np.int64),
    )

    book = load_bad_action_book(str(path))
    same_position = chess.Board(" ".join(board.fen().split()[:4]) + " 7 42")

    assert book is not None
    assert bad_action in book[position_key(same_position)]


def test_good_action_book_loads_actions_and_policy_top_k(tmp_path) -> None:
    board = chess.Board()
    e4_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    d4_action = move_to_action(chess.Move.from_uci("d2d4"), board)
    c4_action = move_to_action(chess.Move.from_uci("c2c4"), board)
    policy = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    policy[0, d4_action] = 0.7
    policy[0, e4_action] = 0.3
    policy[0, c4_action] = 0.5
    action_path = tmp_path / "actions.npz"
    policy_path = tmp_path / "policy.npz"
    combined_path = tmp_path / "combined.npz"
    np.savez_compressed(
        action_path,
        fens=np.asarray([board.fen()]),
        actions=np.asarray([e4_action], dtype=np.int64),
    )
    np.savez_compressed(
        policy_path,
        fens=np.asarray([board.fen()]),
        policies=policy,
    )
    np.savez_compressed(
        combined_path,
        fens=np.asarray([board.fen()]),
        actions=np.asarray([e4_action], dtype=np.int64),
        policies=policy,
    )

    action_book = load_good_action_book(str(action_path))
    policy_book = load_good_action_book(str(policy_path), policy_top_k=1)
    combined_default_book = load_good_action_book(str(combined_path))
    combined_topk_book = load_good_action_book(str(combined_path), policy_top_k=2)

    assert action_book is not None
    assert policy_book is not None
    assert combined_default_book is not None
    assert combined_topk_book is not None
    assert action_book[position_key(board)] == frozenset({e4_action})
    assert policy_book[position_key(board)] == frozenset({d4_action})
    assert combined_default_book[position_key(board)] == frozenset({e4_action})
    assert combined_topk_book[position_key(board)] == frozenset({e4_action, d4_action, c4_action})


def test_mcts_policy_prior_temperature_flattens_priors() -> None:
    board = chess.Board()
    evaluator = SparsePolicyEvaluator({"e2e4": 0.99, "d2d4": 0.01})

    default_search = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(simulations=0, root_mate_search_plies=0),
    )
    flat_search = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(
            simulations=0,
            policy_prior_temperature=2.0,
            root_mate_search_plies=0,
        ),
    )

    default_result = default_search.run(board)
    flat_result = flat_search.run(board)
    e4_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    d4_action = move_to_action(chess.Move.from_uci("d2d4"), board)

    assert default_result.root.children[d4_action].prior < 0.02
    assert flat_result.root.children[d4_action].prior > 0.08
    assert flat_result.root.children[e4_action].prior > flat_result.root.children[d4_action].prior


def test_mcts_root_tactical_prior_can_override_policy_prior() -> None:
    board = chess.Board("q3k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    quiet_action = move_to_action(chess.Move.from_uci("e1d1"), board)
    capture_action = move_to_action(chess.Move.from_uci("a1a8"), board)
    evaluator = SparsePolicyEvaluator({"e1d1": 0.99, "a1a8": 0.01})

    policy_only = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(simulations=0, root_mate_search_plies=0),
    ).run(board)
    tactical = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_tactical_prior_weight=1.0,
            root_tactical_prior_temperature_cp=100.0,
        ),
    ).run(board)

    assert policy_only.select_action(temperature=0.0, rng=np.random.default_rng(0)) == quiet_action
    assert tactical.select_action(temperature=0.0, rng=np.random.default_rng(0)) == capture_action
    assert tactical.root.children[capture_action].prior > tactical.root.children[quiet_action].prior


def test_search_result_breaks_visit_ties_by_prior() -> None:
    board = chess.Board()
    e4_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    d4_action = move_to_action(chess.Move.from_uci("d2d4"), board)
    root = Node(prior=1.0)
    root.children = {
        e4_action: Node(prior=0.1),
        d4_action: Node(prior=0.9),
    }
    visits = np.zeros(ACTION_SIZE, dtype=np.float32)
    result = SearchResult(root=root, visits=visits, root_value=0.0)

    assert result.select_action(temperature=0.0, rng=np.random.default_rng(0)) == d4_action
    assert result.policy(temperature=0.0)[d4_action] == 1.0

    visits[e4_action] = 3
    visits[d4_action] = 3
    result = SearchResult(root=root, visits=visits, root_value=0.0)

    assert result.select_action(temperature=0.0, rng=np.random.default_rng(0)) == d4_action
    assert result.policy(temperature=0.0)[d4_action] == 1.0

    visits[e4_action] = 4
    result = SearchResult(root=root, visits=visits, root_value=0.0)

    assert result.select_action(temperature=0.0, rng=np.random.default_rng(0)) == e4_action


def test_mcts_rejects_invalid_policy_prior_temperature() -> None:
    for temperature in (0.0, -1.0, float("nan")):
        try:
            MCTSConfig(policy_prior_temperature=temperature)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid policy prior temperature was accepted")


def test_mcts_rejects_invalid_leaf_material_weight() -> None:
    for weight in (-0.1, 1.1, float("nan")):
        try:
            MCTSConfig(leaf_material_value_weight=weight)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid leaf material weight was accepted")


def test_mcts_rejects_invalid_root_tactical_prior_settings() -> None:
    for weight in (-0.1, 1.1, float("nan")):
        try:
            MCTSConfig(root_tactical_prior_weight=weight)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid root tactical prior weight was accepted")

    for temperature in (0.0, -1.0, float("nan")):
        try:
            MCTSConfig(root_tactical_prior_temperature_cp=temperature)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid root tactical prior temperature was accepted")


def test_mcts_leaf_material_value_blend_scores_leaf_from_side_to_move() -> None:
    board = chess.Board("7k/8/8/8/8/8/6q1/Q6K w - - 0 1")

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=1,
            root_mate_search_plies=0,
            leaf_material_value_weight=1.0,
            leaf_material_search_plies=0,
        ),
    )
    result = search.run(board)
    action = move_to_action(chess.Move.from_uci("h1g2"), board)

    assert result.root.children[action].value < -0.5
    assert result.root_value > 0.5


def test_mcts_root_prioritizes_mate_in_one() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2")
    mate_action = move_to_action(chess.Move.from_uci("d8h4"), board)

    search = AlphaZeroMCTS(UniformEvaluator(), MCTSConfig(simulations=4))
    result = search.run(board)
    action = result.select_action(temperature=0.0, rng=search.rng)

    assert action == mate_action
    assert set(result.root.children) == {mate_action}


def test_mcts_root_prunes_moves_allowing_mate_in_one() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
    blunder_action = move_to_action(chess.Move.from_uci("g2g4"), board)

    search = AlphaZeroMCTS(UniformEvaluator(), MCTSConfig(simulations=4))
    result = search.run(board)

    assert blunder_action not in result.root.children
    assert result.visits[blunder_action] == 0


def test_mate_search_follows_capture_starting_forced_mate_from_gate_loss() -> None:
    board = chess.Board("3r4/2p4p/1p2k2P/2p1p3/p7/1R2p3/1P3P2/1K4n1 b - - 0 39")

    assert _side_to_move_can_force_mate(board, 5, {})


def test_mcts_root_prunes_deeper_capture_forced_mate_from_gate_loss() -> None:
    board = chess.Board("3r4/2p4p/1p2k2P/2p1p3/p1N2p2/Rb6/1P3P2/1K4n1 w - - 0 38")
    blunder_action = move_to_action(chess.Move.from_uci("c4e3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(simulations=0, root_mate_search_plies=7),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_prunes_large_material_blunder() -> None:
    board = chess.Board("3r2k1/8/8/8/8/8/3N4/3Q2K1 w - - 0 1")
    blunder_action = move_to_action(chess.Move.from_uci("d2f3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=1,
            root_material_max_loss_cp=350,
        ),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_material_zero_max_loss_is_strict() -> None:
    board = chess.Board("3r2k1/8/8/8/8/8/3N4/3Q2K1 w - - 0 1")
    blunder_action = move_to_action(chess.Move.from_uci("d2f3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=1,
            root_material_max_loss_cp=0,
        ),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_material_filter_can_be_disabled() -> None:
    board = chess.Board("3r2k1/8/8/8/8/8/3N4/3Q2K1 w - - 0 1")
    blunder_action = move_to_action(chess.Move.from_uci("d2f3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=0,
        ),
    )
    result = search.run(board)

    assert blunder_action in result.root.children


def test_mcts_root_material_prunes_missed_material_gain_from_gate_loss() -> None:
    board = chess.Board("r1bqkb1r/p1p2ppp/2p5/8/4n3/8/PPP1QPPP/RNB1K2R w KQkq - 0 9")
    castle_action = move_to_action(chess.Move.from_uci("e1g1"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"e1g1": 0.99, "b1c3": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=1,
            root_material_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert castle_action not in result.root.children
    assert result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) != castle_action


def test_material_tactical_moves_are_capped_by_priority() -> None:
    board = chess.Board("8/PPPPPPPP/8/8/4k3/8/8/4K3 w - - 0 1")
    raw_tactical = [
        move
        for move in board.legal_moves
        if board.is_capture(move) or move.promotion or board.gives_check(move)
    ]

    moves = _material_tactical_moves(board)

    assert len(raw_tactical) > TACTICAL_MATERIAL_CANDIDATE_LIMIT
    assert len(moves) == TACTICAL_MATERIAL_CANDIDATE_LIMIT
    assert all(move.promotion for move in moves)


def test_king_safety_candidates_include_quiet_pressure_with_captures() -> None:
    board = chess.Board("2kr1b1r/p1p2ppp/2pnb3/4B3/8/2N5/PPP1RPPP/R5K1 b - - 4 14")
    quiet_pressure = chess.Move.from_uci("e6c4")

    assert quiet_pressure in board.legal_moves
    assert any(
        board.is_capture(move) or move.promotion or board.gives_check(move)
        for move in board.legal_moves
    )
    assert quiet_pressure in _static_safety_candidate_moves(board)


def test_mcts_root_prunes_queen_for_rook_trap_from_stockfish_game() -> None:
    board = chess.Board("r1b1kbnr/p4ppp/2p1p3/2pp4/2P1PB1P/3P1N2/Pq1N1PP1/R2QK2R b KQkq - 0 9")
    blunder_action = move_to_action(chess.Move.from_uci("b2a1"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=2,
            root_material_max_loss_cp=250,
        ),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_material_filter_keeps_best_moves_when_all_moves_lose_material() -> None:
    board = chess.Board("r1bqk2r/1pNp1ppp/p1n1pn2/8/1b2PB2/2N5/PPP2PPP/R2QKB1R b KQkq - 1 8")
    blunder_action = move_to_action(chess.Move.from_uci("d8c7"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=1,
            root_material_max_loss_cp=150,
        ),
    )
    result = search.run(board)

    assert result.root.children
    assert blunder_action not in result.root.children


def test_mcts_root_material_prunes_speculative_checking_capture_from_gate_loss() -> None:
    board = chess.Board("r2q1rk1/p1p2ppp/2P2n2/8/1b4b1/2NB4/PPP2PPP/R1BQK2R w KQ - 1 10")
    sacrifice_action = move_to_action(chess.Move.from_uci("d3h7"), board)
    quiet_action = move_to_action(chess.Move.from_uci("d1d2"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert quiet_action in result.root.children


def test_mcts_root_material_filters_later_speculative_checking_capture_gate_loss() -> None:
    board = chess.Board("r1b1r1k1/p1p2ppp/2q5/8/2P5/2nB1P2/P1P3PP/1RBQ1K1R w - - 1 15")
    sacrifice_action = move_to_action(chess.Move.from_uci("d3h7"), board)
    quiet_action = move_to_action(chess.Move.from_uci("c1d2"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"d3h7": 0.99, "c1d2": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert quiet_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) != sacrifice_action


def test_mcts_good_action_book_can_keep_speculative_checking_capture() -> None:
    board = chess.Board("r1b1r1k1/p1p2ppp/2q5/8/2P5/2nB1P2/P1P3PP/1RBQ1K1R w - - 1 15")
    sacrifice_action = move_to_action(chess.Move.from_uci("d3h7"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_good_action_book={position_key(board): frozenset({sacrifice_action})},
        ),
    )
    result = search.run(board)

    assert set(result.root.children) == {sacrifice_action}


def test_mcts_root_king_safety_runs_before_material_fallback_from_gate_loss() -> None:
    board = chess.Board("r1bqk2r/2p2ppp/p1n5/1p1B4/1b1pn3/2N2N2/PPP2PPP/R1BQR1K1 b kq - 0 10")
    king_walk_action = move_to_action(chess.Move.from_uci("e8d7"), board)
    castle_action = move_to_action(chess.Move.from_uci("e8g8"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"e8d7": 0.99, "e8g8": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert king_walk_action not in result.root.children
    assert castle_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) == castle_action


def test_mcts_speculative_checking_capture_filter_runs_before_king_safety_gate_loss() -> None:
    board = chess.Board("r1b2rk1/ppp2ppp/3b1q2/3P4/2BP1P2/4n3/PPQ3PP/RN3RK1 w - - 0 13")
    sacrifice_action = move_to_action(chess.Move.from_uci("c2h7"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"c2h7": 0.99, "b1c3": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert result.root.children
    assert sacrifice_action not in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) != sacrifice_action


def test_mcts_material_guard_evaluates_before_king_safety_fallback_gate_loss() -> None:
    board = chess.Board("r1bqkb1r/2pp1ppp/p1n5/1p2p3/3Pn3/1B3N2/PPP2PPP/RNBQ1RK1 b kq - 0 7")
    sacrifice_action = move_to_action(chess.Move.from_uci("e4f2"), board)
    safe_action = move_to_action(chess.Move.from_uci("c6d4"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"e4f2": 0.99, "c6d4": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert safe_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) != sacrifice_action


def test_mcts_material_safe_set_vetoes_king_safe_exchange_sacrifice_gate_loss() -> None:
    board = chess.Board("4k1nr/1pp3p1/p1p2pp1/2b1P3/3r2Pq/2NPBQ1P/PPP2PK1/R4R2 b k - 1 15")
    sacrifice_action = move_to_action(chess.Move.from_uci("d4g4"), board)
    safe_action = move_to_action(chess.Move.from_uci("f6e5"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"d4g4": 0.99, "f6e5": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert safe_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) == safe_action


def test_mcts_material_fallback_vetoes_king_fallback_sacrifice_gate_loss() -> None:
    board = chess.Board("r2qkb1r/1npp1ppp/p7/1p2p3/3Pn3/5N2/PPP2PPP/RNBQR1K1 b kq - 1 10")
    sacrifice_action = move_to_action(chess.Move.from_uci("e4f2"), board)
    safe_action = move_to_action(chess.Move.from_uci("f7f5"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"e4f2": 0.99, "f7f5": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert safe_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) == safe_action


def test_mcts_root_material_prunes_king_recapturable_quiet_check_gate_loss() -> None:
    board = chess.Board("3rr1k1/p1p2pp1/2P2q1p/7b/1b1N4/3BnP2/PPP2BPP/R2Q2KR w - - 8 18")
    sacrifice_action = move_to_action(chess.Move.from_uci("d3h7"), board)
    safe_action = move_to_action(chess.Move.from_uci("d1b1"), board)

    search = AlphaZeroMCTS(
        SparsePolicyEvaluator({"d3h7": 0.99, "d1b1": 0.01}),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
            root_king_safety_search_plies=2,
            root_king_safety_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert sacrifice_action not in result.root.children
    assert safe_action in result.root.children
    assert result.select_action(temperature=0.0, rng=search.rng) == safe_action


def test_mcts_root_material_filter_uses_worst_shallow_depth() -> None:
    board = chess.Board(
        "rnb1kbnr/1p1p1ppp/p3p3/1N2q3/8/4BN2/PPP2PPP/R2QKB1R b KQkq - 1 8"
    )
    blunder_action = move_to_action(chess.Move.from_uci("e5b5"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=3,
            root_material_max_loss_cp=100,
        ),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_king_safety_filter_prunes_static_drop() -> None:
    board = chess.Board("3r2k1/8/8/8/8/8/3N4/3Q2K1 w - - 0 1")
    blunder_action = move_to_action(chess.Move.from_uci("d2f3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=0,
            root_king_safety_search_plies=1,
            root_king_safety_max_loss_cp=350,
        ),
    )
    result = search.run(board)

    assert blunder_action not in result.root.children


def test_mcts_root_king_safety_filter_penalizes_castled_pawn_shelter() -> None:
    board = chess.Board(
        "2kr2r1/1pp1qppp/p1pbbn2/4p3/P3P3/3P1N2/1PPN1PPP/R1BQR1K1 w - - 3 11"
    )
    shelter_move = move_to_action(chess.Move.from_uci("h2h3"), board)

    search = AlphaZeroMCTS(
        UniformEvaluator(),
        MCTSConfig(
            simulations=0,
            root_mate_search_plies=0,
            root_material_search_plies=0,
            root_king_safety_search_plies=1,
            root_king_safety_max_loss_cp=50,
        ),
    )
    result = search.run(board)

    assert shelter_move not in result.root.children
