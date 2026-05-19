"""Checkpoint evaluation games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
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
    engine_path: str = "stockfish"
    engine_time: float = 0.05
    engine_depth: int | None = None
    device: str = "auto"
    material_value_weight: float = 0.0
    material_value_search_plies: int = 0
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250
    seed: int = 0
    max_plies: int = 512
    pgn_out: str | None = None


@dataclass
class EvalGameRecord:
    board: chess.Board
    model_color: chess.Color
    score: float
    opponent_name: str


def evaluate_checkpoint(config: EvalConfig) -> dict[str, float]:
    model_eval = load_evaluator(
        config.checkpoint,
        device=config.device,
        material_value_weight=config.material_value_weight,
        material_value_search_plies=config.material_value_search_plies,
    )
    if config.opponent in {"uci", "stockfish"}:
        return evaluate_against_engine(config, model_eval)

    opponent_eval: Evaluator
    if config.opponent_checkpoint:
        opponent_eval = load_evaluator(
            config.opponent_checkpoint,
            device=config.device,
            material_value_weight=config.material_value_weight,
            material_value_search_plies=config.material_value_search_plies,
        )
    else:
        opponent_eval = UniformEvaluator()
    return evaluate_match(
        model_eval=model_eval,
        opponent_eval=opponent_eval,
        games=config.games,
        simulations=config.simulations,
        root_material_search_plies=config.root_material_search_plies,
        root_material_max_loss_cp=config.root_material_max_loss_cp,
        seed=config.seed,
        max_plies=config.max_plies,
        pgn_out=config.pgn_out,
        opponent_name=config.opponent_checkpoint or config.opponent,
    )


def evaluate_against_engine(config: EvalConfig, model_eval: Evaluator) -> dict[str, float]:
    rng = np.random.default_rng(config.seed)
    scores: list[float] = []
    records: list[EvalGameRecord] = []
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)
    with chess.engine.SimpleEngine.popen_uci(config.engine_path) as engine:
        for game_idx in range(config.games):
            model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            score, board = play_eval_game_against_engine(
                model_eval=model_eval,
                engine=engine,
                model_color=model_color,
                simulations=config.simulations,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
                max_plies=config.max_plies,
                limit=limit,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
            scores.append(score)
            records.append(
                EvalGameRecord(
                    board=board,
                    model_color=model_color,
                    score=score,
                    opponent_name=config.engine_path,
                )
            )
    if config.pgn_out:
        write_eval_pgns(config.pgn_out, records)
    return summarize_scores(scores)


def evaluate_match(
    model_eval: Evaluator,
    opponent_eval: Evaluator,
    games: int,
    simulations: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    seed: int,
    max_plies: int,
    pgn_out: str | None = None,
    opponent_name: str = "opponent",
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    records: list[EvalGameRecord] = []

    for game_idx in range(games):
        model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        score, board = play_eval_game(
            model_eval,
            opponent_eval,
            model_color,
            simulations=simulations,
            root_material_search_plies=root_material_search_plies,
            root_material_max_loss_cp=root_material_max_loss_cp,
            max_plies=max_plies,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        scores.append(score)
        records.append(
            EvalGameRecord(
                board=board,
                model_color=model_color,
                score=score,
                opponent_name=opponent_name,
            )
        )

    if pgn_out:
        write_eval_pgns(pgn_out, records)
    return summarize_scores(scores)


def play_eval_game(
    model_eval: Evaluator,
    opponent_eval: Evaluator,
    model_color: chess.Color,
    simulations: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    max_plies: int,
    rng: np.random.Generator,
) -> tuple[float, chess.Board]:
    board = chess.Board()
    mcts_config = MCTSConfig(
        simulations=simulations,
        root_material_search_plies=root_material_search_plies,
        root_material_max_loss_cp=root_material_max_loss_cp,
    )
    model_mcts = AlphaZeroMCTS(model_eval, mcts_config, rng=rng)
    opponent_mcts = AlphaZeroMCTS(opponent_eval, mcts_config, rng=rng)

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
    score = 1.0 if value > 0 else 0.5 if value == 0 else 0.0
    return score, board


def play_eval_game_against_engine(
    model_eval: Evaluator,
    engine: chess.engine.SimpleEngine,
    model_color: chess.Color,
    simulations: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    max_plies: int,
    limit: chess.engine.Limit,
    rng: np.random.Generator,
) -> tuple[float, chess.Board]:
    board = chess.Board()
    model_mcts = AlphaZeroMCTS(
        model_eval,
        MCTSConfig(
            simulations=simulations,
            root_material_search_plies=root_material_search_plies,
            root_material_max_loss_cp=root_material_max_loss_cp,
        ),
        rng=rng,
    )

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == model_color:
            result = model_mcts.run(board)
            action = result.select_action(temperature=0.0, rng=rng)
            if action is None:
                break
            move = action_to_move(action, board)
            if move is None:
                raise RuntimeError(f"Model selected illegal action {action} in {board.fen()}")
        else:
            move = engine.play(board, limit).move
        board.push(move)

    value = result_value_for_color(board, model_color)
    score = 1.0 if value > 0 else 0.5 if value == 0 else 0.0
    return score, board


def write_eval_pgns(path: str | Path, records: list[EvalGameRecord]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            game = chess.pgn.Game.from_board(record.board)
            result = record.board.result(claim_draw=True)
            if result == "*" and record.score == 0.5:
                result = "1/2-1/2"
            game.headers["Event"] = "AlphaChess evaluation"
            game.headers["Round"] = str(index)
            game.headers["White"] = (
                "AlphaChess" if record.model_color == chess.WHITE else record.opponent_name
            )
            game.headers["Black"] = (
                record.opponent_name if record.model_color == chess.WHITE else "AlphaChess"
            )
            game.headers["Result"] = result
            game.headers["AlphaChessScore"] = str(record.score)
            print(game, file=handle, end="\n\n")
    return output


def summarize_scores(scores: list[float]) -> dict[str, float]:
    return {
        "games": float(len(scores)),
        "score": float(sum(scores)),
        "score_rate": float(sum(scores) / max(1, len(scores))),
        "wins": float(sum(1 for score in scores if score == 1.0)),
        "draws": float(sum(1 for score in scores if score == 0.5)),
        "losses": float(sum(1 for score in scores if score == 0.0)),
    }
