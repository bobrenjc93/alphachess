"""Self-play data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

import chess
import numpy as np

from alpha_chess.chess_env import action_to_move, encode_board, result_value_for_color
from alpha_chess.evaluator import Evaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig


@dataclass
class SelfPlayConfig:
    games: int = 1
    simulations: int = 64
    max_plies: int = 512
    temperature_moves: int = 20
    seed: int = 0


def play_game(evaluator: Evaluator, config: SelfPlayConfig, game_seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(game_seed)
    board = chess.Board()
    boards: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    turns: list[chess.Color] = []
    fens: list[str] = []
    moves: list[str] = []

    mcts = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(simulations=config.simulations, add_root_noise=True),
        rng=rng,
    )

    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        result = mcts.run(board)
        temperature = 1.0 if ply < config.temperature_moves else 0.0
        policy = result.policy(temperature)
        action = result.select_action(temperature, rng)
        if action is None:
            break
        move = action_to_move(action, board)
        if move is None:
            raise RuntimeError(f"MCTS selected illegal action {action} in {board.fen()}")

        boards.append(encode_board(board))
        policies.append(policy)
        turns.append(board.turn)
        fens.append(board.fen())
        moves.append(move.uci())
        board.push(move)

    white_value = result_value_for_color(board, chess.WHITE)
    values = np.asarray(
        [white_value if turn == chess.WHITE else -white_value for turn in turns],
        dtype=np.float32,
    )

    return {
        "boards": np.asarray(boards, dtype=np.float32),
        "policies": np.asarray(policies, dtype=np.float32),
        "values": values,
        "moves": np.asarray(moves),
        "fens": np.asarray(fens),
        "result": np.asarray(board.result(claim_draw=True)),
        "final_fen": np.asarray(board.fen()),
    }


def generate_self_play(
    evaluator: Evaluator,
    out_dir: str | Path,
    config: SelfPlayConfig,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    start = time()
    for game_idx in range(config.games):
        game = play_game(evaluator, config, config.seed + game_idx)
        stamp = int(start)
        path = out / f"game_{stamp}_{game_idx:06d}.npz"
        np.savez_compressed(path, **game)
        written.append(path)
    return written
