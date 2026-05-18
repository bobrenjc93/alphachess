import chess

from alpha_chess.chess_env import action_to_move
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
