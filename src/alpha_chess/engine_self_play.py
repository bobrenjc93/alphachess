"""Generate UCI-engine self-play PGNs for teacher data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import chess
import chess.engine
import chess.pgn
import numpy as np


@dataclass
class EngineSelfPlayConfig:
    out: str = "data/engine_selfplay/stockfish_selfplay.pgn"
    engine_path: str = "stockfish"
    white_engine_path: str | None = None
    black_engine_path: str | None = None
    engine_time: float = 0.05
    engine_depth: int | None = None
    games: int = 1
    max_plies: int = 200
    opening_random_plies: int = 0
    opening_multipv: int = 4
    opening_temperature_cp: float = 80.0
    seed: int = 0
    event: str = "AlphaChess engine self-play"


def generate_engine_self_play(config: EngineSelfPlayConfig) -> Path:
    """Write engine-vs-engine games as PGN and return the PGN path."""

    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if config.engine_time <= 0 and config.engine_depth is None:
        raise ValueError("engine_time must be positive unless engine_depth is set")
    if config.opening_multipv <= 0:
        raise ValueError("opening_multipv must be positive")
    if config.opening_temperature_cp < 0:
        raise ValueError("opening_temperature_cp must be non-negative")

    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)
    white_path = config.white_engine_path or config.engine_path
    black_path = config.black_engine_path or config.engine_path
    rng = np.random.default_rng(config.seed)
    generated_at = datetime.now(timezone.utc)
    outcomes: list[str] = []
    ply_counts: list[int] = []

    with _engine_pair(white_path, black_path) as (white_engine, black_engine):
        with out_path.open("w", encoding="utf-8") as handle:
            for game_index in range(config.games):
                game, result, ply_count = _play_engine_game(
                    white_engine=white_engine,
                    black_engine=black_engine,
                    limit=limit,
                    rng=rng,
                    config=config,
                    game_index=game_index,
                    generated_at=generated_at,
                    white_path=white_path,
                    black_path=black_path,
                )
                outcomes.append(result)
                ply_counts.append(ply_count)
                print(game, file=handle, end="\n\n")

    _write_summary(out_path, config, generated_at, outcomes, ply_counts)
    return out_path


class _engine_pair:
    def __init__(self, white_path: str, black_path: str) -> None:
        self.white_path = white_path
        self.black_path = black_path
        self.white_engine: chess.engine.SimpleEngine | None = None
        self.black_engine: chess.engine.SimpleEngine | None = None

    def __enter__(
        self,
    ) -> tuple[chess.engine.SimpleEngine, chess.engine.SimpleEngine]:
        self.white_engine = chess.engine.SimpleEngine.popen_uci(self.white_path)
        if self.black_path == self.white_path:
            self.black_engine = self.white_engine
        else:
            self.black_engine = chess.engine.SimpleEngine.popen_uci(self.black_path)
        return self.white_engine, self.black_engine

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.black_engine is not None and self.black_engine is not self.white_engine:
            self.black_engine.quit()
        if self.white_engine is not None:
            self.white_engine.quit()


def _play_engine_game(
    white_engine: chess.engine.SimpleEngine,
    black_engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    rng: np.random.Generator,
    config: EngineSelfPlayConfig,
    game_index: int,
    generated_at: datetime,
    white_path: str,
    black_path: str,
) -> tuple[chess.pgn.Game, str, int]:
    game = chess.pgn.Game()
    game.headers["Event"] = config.event
    game.headers["Site"] = "AlphaChess"
    game.headers["Date"] = generated_at.strftime("%Y.%m.%d")
    game.headers["UTCDate"] = generated_at.strftime("%Y.%m.%d")
    game.headers["UTCTime"] = generated_at.strftime("%H:%M:%S")
    game.headers["Round"] = str(game_index + 1)
    game.headers["White"] = _engine_name(white_path)
    game.headers["Black"] = _engine_name(black_path)
    game.headers["WhiteEnginePath"] = white_path
    game.headers["BlackEnginePath"] = black_path

    board = game.board()
    node: chess.pgn.GameNode = game
    termination = "Normal"
    for ply in range(config.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = _select_engine_move(
            engine=engine,
            board=board,
            limit=limit,
            rng=rng,
            randomize=ply < config.opening_random_plies,
            multipv=config.opening_multipv,
            temperature_cp=config.opening_temperature_cp,
        )
        if move is None or move not in board.legal_moves:
            termination = "No legal engine move"
            break
        node = node.add_variation(move)
        board.push(move)
    else:
        if not board.is_game_over(claim_draw=True):
            termination = "Max plies"

    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"
    game.headers["Result"] = result
    game.headers["PlyCount"] = str(board.ply())
    game.headers["Termination"] = termination
    return game, result, board.ply()


def _select_engine_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    rng: np.random.Generator,
    randomize: bool,
    multipv: int,
    temperature_cp: float,
) -> chess.Move | None:
    if not randomize or multipv <= 1:
        return engine.play(board, limit).move

    infos = _analyse_multipv(engine, board, limit, multipv)
    scored_moves = _scored_moves_from_infos(board, infos)
    if not scored_moves:
        return engine.play(board, limit).move
    if temperature_cp <= 0 or len(scored_moves) == 1:
        return max(scored_moves, key=lambda item: item[1])[0]

    scores = np.asarray([score for _, score in scored_moves], dtype=np.float64)
    scores -= float(np.max(scores))
    probabilities = np.exp(scores / float(temperature_cp))
    total = float(probabilities.sum())
    if total <= 0 or not np.isfinite(total):
        return max(scored_moves, key=lambda item: item[1])[0]
    probabilities /= total
    index = int(rng.choice(np.arange(len(scored_moves)), p=probabilities))
    return scored_moves[index][0]


def _analyse_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    multipv: int,
) -> list[dict]:
    analysis = engine.analyse(board, limit, multipv=max(1, multipv))
    return analysis if isinstance(analysis, list) else [analysis]


def _scored_moves_from_infos(
    board: chess.Board,
    infos: list[dict],
) -> list[tuple[chess.Move, float]]:
    scores_by_move: dict[chess.Move, float] = {}
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        centipawns = score.pov(board.turn).score(mate_score=100_000)
        if centipawns is None:
            continue
        scores_by_move[move] = max(float(centipawns), scores_by_move.get(move, -float("inf")))
    return sorted(scores_by_move.items(), key=lambda item: item[1], reverse=True)


def _engine_name(path: str) -> str:
    name = Path(path).name
    return name or "UCIEngine"


def _write_summary(
    out_path: Path,
    config: EngineSelfPlayConfig,
    generated_at: datetime,
    outcomes: list[str],
    ply_counts: list[int],
) -> Path:
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.txt")
    result_counts = {result: outcomes.count(result) for result in sorted(set(outcomes))}
    average_plies = float(np.mean(ply_counts)) if ply_counts else 0.0
    summary_path.write_text(
        "\n".join(
            [
                f"generated_at={generated_at.isoformat()}",
                f"out={out_path}",
                f"games={len(outcomes)}",
                f"result_counts={result_counts}",
                f"average_plies={average_plies:.2f}",
                f"config={asdict(config)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path
