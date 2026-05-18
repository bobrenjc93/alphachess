import chess

from alpha_chess.chess_env import action_to_move, move_to_action
from alpha_chess.evaluator import UniformEvaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig


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
