"""Checkpoint evaluation games."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from alpha_chess.chess_env import action_to_move, result_value_for_color
from alpha_chess.evaluator import Evaluator, UniformEvaluator, load_evaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig


@dataclass
class EvalConfig:
    checkpoint: str
    games: int = 2
    simulations: int = 64
    opponent: str = "uniform"
    opponent_checkpoint: str | None = None
    device: str = "auto"
    seed: int = 0
    max_plies: int = 512


def evaluate_checkpoint(config: EvalConfig) -> dict[str, float]:
    model_eval = load_evaluator(config.checkpoint, device=config.device)
    opponent_eval: Evaluator
    if config.opponent_checkpoint:
        opponent_eval = load_evaluator(config.opponent_checkpoint, device=config.device)
    else:
        opponent_eval = UniformEvaluator()
    return evaluate_match(
        model_eval=model_eval,
        opponent_eval=opponent_eval,
        games=config.games,
        simulations=config.simulations,
        seed=config.seed,
        max_plies=config.max_plies,
    )


def evaluate_match(
    model_eval: Evaluator,
    opponent_eval: Evaluator,
    games: int,
    simulations: int,
    seed: int,
    max_plies: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for game_idx in range(games):
        model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        score = play_eval_game(
            model_eval,
            opponent_eval,
            model_color,
            simulations=simulations,
            max_plies=max_plies,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        scores.append(score)

    return {
        "games": float(games),
        "score": float(sum(scores)),
        "score_rate": float(sum(scores) / max(1, len(scores))),
        "wins": float(sum(1 for score in scores if score == 1.0)),
        "draws": float(sum(1 for score in scores if score == 0.5)),
        "losses": float(sum(1 for score in scores if score == 0.0)),
    }


def play_eval_game(
    model_eval: Evaluator,
    opponent_eval: Evaluator,
    model_color: chess.Color,
    simulations: int,
    max_plies: int,
    rng: np.random.Generator,
) -> float:
    board = chess.Board()
    model_mcts = AlphaZeroMCTS(model_eval, MCTSConfig(simulations=simulations), rng=rng)
    opponent_mcts = AlphaZeroMCTS(opponent_eval, MCTSConfig(simulations=simulations), rng=rng)

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        mcts = model_mcts if board.turn == model_color else opponent_mcts
        result = mcts.run(board)
        action = result.select_action(temperature=0.0, rng=rng)
        if action is None:
            break
        move = action_to_move(action, board)
        if move is None:
            raise RuntimeError(f"Evaluator selected illegal action {action} in {board.fen()}")
        board.push(move)

    value = result_value_for_color(board, model_color)
    return 1.0 if value > 0 else 0.5 if value == 0 else 0.0
