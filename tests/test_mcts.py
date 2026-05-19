import chess
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, action_to_move, legal_actions, move_to_action
from alpha_chess.evaluator import UniformEvaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig, Node, SearchResult, advance_root


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
