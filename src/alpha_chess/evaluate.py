"""Checkpoint evaluation games."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from alpha_chess.bad_action_book import (
    BadActionBook,
    GoodActionBook,
    load_bad_action_book,
    load_good_action_book,
)
from alpha_chess.chess_env import action_to_move, move_to_action, result_value_for_color
from alpha_chess.evaluator import Evaluator, UniformEvaluator, load_evaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig, Node, advance_root


@dataclass
class EvalConfig:
    checkpoint: str
    games: int = 2
    simulations: int = 64
    c_puct: float = 1.5
    policy_prior_temperature: float = 1.0
    tree_reuse: bool = True
    opponent: str = "uniform"
    opponent_checkpoint: str | None = None
    engine_path: str = "stockfish"
    engine_time: float = 0.05
    engine_depth: int | None = None
    device: str = "auto"
    material_value_weight: float = 0.0
    material_value_search_plies: int = 0
    leaf_material_value_weight: float = 0.0
    leaf_material_search_plies: int = 0
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250
    root_king_safety_search_plies: int = 0
    root_king_safety_max_loss_cp: int = 250
    root_tactical_prior_weight: float = 0.0
    root_tactical_prior_temperature_cp: float = 200.0
    seed: int = 0
    max_plies: int = 512
    pgn_out: str | None = None
    workers: int = 1
    good_action_book: str | list[str] | None = None
    good_action_book_top_k: int = 1
    bad_action_book: str | list[str] | None = None


@dataclass
class EvalGameRecord:
    board: chess.Board
    model_color: chess.Color
    score: float
    opponent_name: str


def evaluate_checkpoint(config: EvalConfig) -> dict[str, float]:
    if config.workers > 1 and config.games > 1:
        return evaluate_checkpoint_parallel(config)

    model_eval = load_evaluator(
        config.checkpoint,
        device=config.device,
        material_value_weight=config.material_value_weight,
        material_value_search_plies=config.material_value_search_plies,
    )
    good_action_book = load_good_action_book(
        config.good_action_book,
        policy_top_k=config.good_action_book_top_k,
    )
    bad_action_book = load_bad_action_book(config.bad_action_book)
    if config.opponent in {"uci", "stockfish"}:
        return evaluate_against_engine(
            config,
            model_eval,
            good_action_book=good_action_book,
            bad_action_book=bad_action_book,
        )

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
        c_puct=config.c_puct,
        policy_prior_temperature=config.policy_prior_temperature,
        tree_reuse=config.tree_reuse,
        root_mate_search_plies=config.root_mate_search_plies,
        root_material_search_plies=config.root_material_search_plies,
        root_material_max_loss_cp=config.root_material_max_loss_cp,
        root_king_safety_search_plies=config.root_king_safety_search_plies,
        root_king_safety_max_loss_cp=config.root_king_safety_max_loss_cp,
        root_tactical_prior_weight=config.root_tactical_prior_weight,
        root_tactical_prior_temperature_cp=config.root_tactical_prior_temperature_cp,
        leaf_material_value_weight=config.leaf_material_value_weight,
        leaf_material_search_plies=config.leaf_material_search_plies,
        good_action_book=good_action_book,
        bad_action_book=bad_action_book,
        seed=config.seed,
        max_plies=config.max_plies,
        pgn_out=config.pgn_out,
        opponent_name=config.opponent_checkpoint or config.opponent,
    )


def evaluate_checkpoint_parallel(config: EvalConfig) -> dict[str, float]:
    workers = min(max(1, int(config.workers)), max(1, int(config.games)))
    game_seeds = _game_seeds(config.seed, config.games)
    tasks = [(config, game_idx, game_seeds[game_idx]) for game_idx in range(config.games)]
    worker = (
        _evaluate_engine_game_task
        if config.opponent in {"uci", "stockfish"}
        else _evaluate_match_game_task
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        results = list(executor.map(worker, tasks))

    results.sort(key=lambda result: result[0])
    scores = [score for _, score, _ in results]
    records = [record for _, _, record in results]
    if config.pgn_out:
        write_eval_pgns(config.pgn_out, records)
    return summarize_scores(scores)


def evaluate_against_engine(
    config: EvalConfig,
    model_eval: Evaluator,
    good_action_book: GoodActionBook | None = None,
    bad_action_book: BadActionBook | None = None,
) -> dict[str, float]:
    game_seeds = _game_seeds(config.seed, config.games)
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
                c_puct=config.c_puct,
                policy_prior_temperature=config.policy_prior_temperature,
                tree_reuse=config.tree_reuse,
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
                root_king_safety_search_plies=config.root_king_safety_search_plies,
                root_king_safety_max_loss_cp=config.root_king_safety_max_loss_cp,
                root_tactical_prior_weight=config.root_tactical_prior_weight,
                root_tactical_prior_temperature_cp=config.root_tactical_prior_temperature_cp,
                leaf_material_value_weight=config.leaf_material_value_weight,
                leaf_material_search_plies=config.leaf_material_search_plies,
                good_action_book=good_action_book,
                bad_action_book=bad_action_book,
                max_plies=config.max_plies,
                limit=limit,
                rng=np.random.default_rng(game_seeds[game_idx]),
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
    c_puct: float,
    policy_prior_temperature: float,
    root_mate_search_plies: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    seed: int,
    max_plies: int,
    pgn_out: str | None = None,
    opponent_name: str = "opponent",
    tree_reuse: bool = True,
    leaf_material_value_weight: float = 0.0,
    leaf_material_search_plies: int = 0,
    root_king_safety_search_plies: int = 0,
    root_king_safety_max_loss_cp: int = 250,
    root_tactical_prior_weight: float = 0.0,
    root_tactical_prior_temperature_cp: float = 200.0,
    good_action_book: GoodActionBook | None = None,
    bad_action_book: BadActionBook | None = None,
) -> dict[str, float]:
    game_seeds = _game_seeds(seed, games)
    scores: list[float] = []
    records: list[EvalGameRecord] = []

    for game_idx in range(games):
        model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        score, board = play_eval_game(
            model_eval,
            opponent_eval,
            model_color,
            simulations=simulations,
            c_puct=c_puct,
            policy_prior_temperature=policy_prior_temperature,
            tree_reuse=tree_reuse,
            root_mate_search_plies=root_mate_search_plies,
            root_material_search_plies=root_material_search_plies,
            root_material_max_loss_cp=root_material_max_loss_cp,
            root_king_safety_search_plies=root_king_safety_search_plies,
            root_king_safety_max_loss_cp=root_king_safety_max_loss_cp,
            root_tactical_prior_weight=root_tactical_prior_weight,
            root_tactical_prior_temperature_cp=root_tactical_prior_temperature_cp,
            leaf_material_value_weight=leaf_material_value_weight,
            leaf_material_search_plies=leaf_material_search_plies,
            good_action_book=good_action_book,
            bad_action_book=bad_action_book,
            max_plies=max_plies,
            rng=np.random.default_rng(game_seeds[game_idx]),
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


def _evaluate_match_game_task(
    task: tuple[EvalConfig, int, int],
) -> tuple[int, float, EvalGameRecord]:
    config, game_idx, game_seed = task
    model_eval = load_evaluator(
        config.checkpoint,
        device=config.device,
        material_value_weight=config.material_value_weight,
        material_value_search_plies=config.material_value_search_plies,
    )
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
    good_action_book = load_good_action_book(
        config.good_action_book,
        policy_top_k=config.good_action_book_top_k,
    )
    bad_action_book = load_bad_action_book(config.bad_action_book)
    model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    score, board = play_eval_game(
        model_eval,
        opponent_eval,
        model_color,
        simulations=config.simulations,
        c_puct=config.c_puct,
        policy_prior_temperature=config.policy_prior_temperature,
        tree_reuse=config.tree_reuse,
        root_mate_search_plies=config.root_mate_search_plies,
        root_material_search_plies=config.root_material_search_plies,
        root_material_max_loss_cp=config.root_material_max_loss_cp,
        root_king_safety_search_plies=config.root_king_safety_search_plies,
        root_king_safety_max_loss_cp=config.root_king_safety_max_loss_cp,
        root_tactical_prior_weight=config.root_tactical_prior_weight,
        root_tactical_prior_temperature_cp=config.root_tactical_prior_temperature_cp,
        leaf_material_value_weight=config.leaf_material_value_weight,
        leaf_material_search_plies=config.leaf_material_search_plies,
        good_action_book=good_action_book,
        bad_action_book=bad_action_book,
        max_plies=config.max_plies,
        rng=np.random.default_rng(game_seed),
    )
    return (
        game_idx,
        score,
        EvalGameRecord(
            board=board,
            model_color=model_color,
            score=score,
            opponent_name=config.opponent_checkpoint or config.opponent,
        ),
    )


def _evaluate_engine_game_task(
    task: tuple[EvalConfig, int, int],
) -> tuple[int, float, EvalGameRecord]:
    config, game_idx, game_seed = task
    model_eval = load_evaluator(
        config.checkpoint,
        device=config.device,
        material_value_weight=config.material_value_weight,
        material_value_search_plies=config.material_value_search_plies,
    )
    good_action_book = load_good_action_book(
        config.good_action_book,
        policy_top_k=config.good_action_book_top_k,
    )
    bad_action_book = load_bad_action_book(config.bad_action_book)
    model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)
    with chess.engine.SimpleEngine.popen_uci(config.engine_path) as engine:
        score, board = play_eval_game_against_engine(
            model_eval=model_eval,
            engine=engine,
            model_color=model_color,
            simulations=config.simulations,
            c_puct=config.c_puct,
            policy_prior_temperature=config.policy_prior_temperature,
            tree_reuse=config.tree_reuse,
            root_mate_search_plies=config.root_mate_search_plies,
            root_material_search_plies=config.root_material_search_plies,
            root_material_max_loss_cp=config.root_material_max_loss_cp,
            root_king_safety_search_plies=config.root_king_safety_search_plies,
            root_king_safety_max_loss_cp=config.root_king_safety_max_loss_cp,
            root_tactical_prior_weight=config.root_tactical_prior_weight,
            root_tactical_prior_temperature_cp=config.root_tactical_prior_temperature_cp,
            leaf_material_value_weight=config.leaf_material_value_weight,
            leaf_material_search_plies=config.leaf_material_search_plies,
            good_action_book=good_action_book,
            bad_action_book=bad_action_book,
            max_plies=config.max_plies,
            limit=limit,
            rng=np.random.default_rng(game_seed),
        )
    return (
        game_idx,
        score,
        EvalGameRecord(
            board=board,
            model_color=model_color,
            score=score,
            opponent_name=config.engine_path,
        ),
    )


def _game_seeds(seed: int, games: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(games)]


def play_eval_game(
    model_eval: Evaluator,
    opponent_eval: Evaluator,
    model_color: chess.Color,
    simulations: int,
    c_puct: float,
    policy_prior_temperature: float,
    root_mate_search_plies: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    max_plies: int,
    rng: np.random.Generator,
    tree_reuse: bool = True,
    leaf_material_value_weight: float = 0.0,
    leaf_material_search_plies: int = 0,
    root_king_safety_search_plies: int = 0,
    root_king_safety_max_loss_cp: int = 250,
    root_tactical_prior_weight: float = 0.0,
    root_tactical_prior_temperature_cp: float = 200.0,
    good_action_book: GoodActionBook | None = None,
    bad_action_book: BadActionBook | None = None,
) -> tuple[float, chess.Board]:
    board = chess.Board()
    mcts_config = MCTSConfig(
        simulations=simulations,
        c_puct=c_puct,
        policy_prior_temperature=policy_prior_temperature,
        root_mate_search_plies=root_mate_search_plies,
        root_material_search_plies=root_material_search_plies,
        root_material_max_loss_cp=root_material_max_loss_cp,
        root_king_safety_search_plies=root_king_safety_search_plies,
        root_king_safety_max_loss_cp=root_king_safety_max_loss_cp,
        root_tactical_prior_weight=root_tactical_prior_weight,
        root_tactical_prior_temperature_cp=root_tactical_prior_temperature_cp,
        leaf_material_value_weight=leaf_material_value_weight,
        leaf_material_search_plies=leaf_material_search_plies,
        root_good_action_book=good_action_book,
        root_bad_action_book=bad_action_book,
    )
    opponent_mcts_config = MCTSConfig(
        simulations=simulations,
        c_puct=c_puct,
        policy_prior_temperature=policy_prior_temperature,
        root_mate_search_plies=root_mate_search_plies,
        root_material_search_plies=root_material_search_plies,
        root_material_max_loss_cp=root_material_max_loss_cp,
        root_king_safety_search_plies=root_king_safety_search_plies,
        root_king_safety_max_loss_cp=root_king_safety_max_loss_cp,
        root_tactical_prior_weight=root_tactical_prior_weight,
        root_tactical_prior_temperature_cp=root_tactical_prior_temperature_cp,
        leaf_material_value_weight=leaf_material_value_weight,
        leaf_material_search_plies=leaf_material_search_plies,
    )
    model_mcts = AlphaZeroMCTS(model_eval, mcts_config, rng=rng)
    opponent_mcts = AlphaZeroMCTS(opponent_eval, opponent_mcts_config, rng=rng)
    model_root: Node | None = None
    opponent_root: Node | None = None

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == model_color:
            result = model_mcts.run(board, root=model_root)
            model_root = result.root
        else:
            result = opponent_mcts.run(board, root=opponent_root)
            opponent_root = result.root
        action = result.select_action(temperature=0.0, rng=rng)
        if action is None:
            break
        move = action_to_move(action, board)
        if move is None:
            raise RuntimeError(f"Evaluator selected illegal action {action} in {board.fen()}")
        if tree_reuse:
            model_root = advance_root(model_root, action)
            opponent_root = advance_root(opponent_root, action)
        else:
            model_root = None
            opponent_root = None
        board.push(move)

    value = result_value_for_color(board, model_color)
    score = 1.0 if value > 0 else 0.5 if value == 0 else 0.0
    return score, board


def play_eval_game_against_engine(
    model_eval: Evaluator,
    engine: chess.engine.SimpleEngine,
    model_color: chess.Color,
    simulations: int,
    c_puct: float,
    policy_prior_temperature: float,
    root_mate_search_plies: int,
    root_material_search_plies: int,
    root_material_max_loss_cp: int,
    max_plies: int,
    limit: chess.engine.Limit,
    rng: np.random.Generator,
    tree_reuse: bool = True,
    leaf_material_value_weight: float = 0.0,
    leaf_material_search_plies: int = 0,
    root_king_safety_search_plies: int = 0,
    root_king_safety_max_loss_cp: int = 250,
    root_tactical_prior_weight: float = 0.0,
    root_tactical_prior_temperature_cp: float = 200.0,
    good_action_book: GoodActionBook | None = None,
    bad_action_book: BadActionBook | None = None,
) -> tuple[float, chess.Board]:
    board = chess.Board()
    model_mcts = AlphaZeroMCTS(
        model_eval,
        MCTSConfig(
            simulations=simulations,
            c_puct=c_puct,
            policy_prior_temperature=policy_prior_temperature,
            root_mate_search_plies=root_mate_search_plies,
            root_material_search_plies=root_material_search_plies,
            root_material_max_loss_cp=root_material_max_loss_cp,
            root_king_safety_search_plies=root_king_safety_search_plies,
            root_king_safety_max_loss_cp=root_king_safety_max_loss_cp,
            root_tactical_prior_weight=root_tactical_prior_weight,
            root_tactical_prior_temperature_cp=root_tactical_prior_temperature_cp,
            leaf_material_value_weight=leaf_material_value_weight,
            leaf_material_search_plies=leaf_material_search_plies,
            root_good_action_book=good_action_book,
            root_bad_action_book=bad_action_book,
        ),
        rng=rng,
    )
    model_root: Node | None = None

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == model_color:
            result = model_mcts.run(board, root=model_root)
            model_root = result.root
            action = result.select_action(temperature=0.0, rng=rng)
            if action is None:
                break
            move = action_to_move(action, board)
            if move is None:
                raise RuntimeError(f"Model selected illegal action {action} in {board.fen()}")
        else:
            move = engine.play(board, limit).move
            action = move_to_action(move, board)
        model_root = advance_root(model_root, action) if tree_reuse else None
        board.push(move)

    value = result_value_for_color(board, model_color)
    score = 1.0 if value > 0 else 0.5 if value == 0 else 0.0
    return score, board


def write_eval_pgns(
    path: str | Path,
    records: list[EvalGameRecord],
    timestamp: datetime | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().astimezone()
    timestamp = timestamp.replace(microsecond=0)
    iso_timestamp = timestamp.isoformat()
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            game = chess.pgn.Game.from_board(record.board)
            result = record.board.result(claim_draw=True)
            if result == "*" and record.score == 0.5:
                result = "1/2-1/2"
            game.headers["Event"] = "AlphaChess evaluation"
            game.headers["Date"] = timestamp.strftime("%Y.%m.%d")
            game.headers["Time"] = timestamp.strftime("%H:%M:%S")
            game.headers["Round"] = str(index)
            game.headers["White"] = (
                "AlphaChess" if record.model_color == chess.WHITE else record.opponent_name
            )
            game.headers["Black"] = (
                record.opponent_name if record.model_color == chess.WHITE else "AlphaChess"
            )
            game.headers["Result"] = result
            game.headers["AlphaChessScore"] = str(record.score)
            game.headers["AlphaChessTimestamp"] = iso_timestamp
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
