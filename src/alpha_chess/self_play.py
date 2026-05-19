"""Self-play data generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250
    seed: int = 0
    workers: int = 1


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
        MCTSConfig(
            simulations=config.simulations,
            add_root_noise=True,
            root_mate_search_plies=config.root_mate_search_plies,
            root_material_search_plies=config.root_material_search_plies,
            root_material_max_loss_cp=config.root_material_max_loss_cp,
        ),
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
    start = time()
    stamp = int(start)
    workers = max(1, int(config.workers))
    if workers == 1 or config.games <= 1:
        written: list[Path] = []
        for game_idx in range(config.games):
            written.append(_play_and_write_game(evaluator, config, out, stamp, game_idx))
        return written

    with ThreadPoolExecutor(max_workers=min(workers, config.games)) as executor:
        return list(
            executor.map(
                lambda game_idx: _play_and_write_game(evaluator, config, out, stamp, game_idx),
                range(config.games),
            )
        )


def _play_and_write_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    out: Path,
    stamp: int,
    game_idx: int,
) -> Path:
    game = play_game(evaluator, config, config.seed + game_idx)
    path = out / f"game_{stamp}_{game_idx:06d}.npz"
    np.savez_compressed(path, **game)
    return path
