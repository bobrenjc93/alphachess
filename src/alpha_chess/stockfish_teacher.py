"""Generate Stockfish teacher labels for chess positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from alpha_chess.chess_env import encode_board, move_to_action
from alpha_chess.pgn_import import PGNImportConfig, _open_pgn_text, _passes_filters


@dataclass
class StockfishTeacherConfig:
    pgn: str
    out: str = "data/teacher/stockfish"
    engine_path: str = "stockfish"
    engine_time: float = 0.02
    engine_depth: int | None = None
    max_games: int | None = None
    max_positions: int = 1024
    min_elo: int | None = None
    min_initial_seconds: int | None = None
    position_stride: int = 4
    chunk_size: int = 1024


def generate_stockfish_teacher(config: StockfishTeacherConfig) -> list[Path]:
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = Path(config.pgn)
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)
    filter_config = PGNImportConfig(
        pgn=config.pgn,
        out=config.out,
        max_games=config.max_games,
        min_elo=config.min_elo,
        min_initial_seconds=config.min_initial_seconds,
    )

    boards: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    fens: list[str] = []
    best_moves: list[str] = []
    written: list[Path] = []
    games_seen = 0
    games_used = 0
    positions = 0

    with chess.engine.SimpleEngine.popen_uci(config.engine_path) as engine:
        with _open_pgn_text(pgn_path) as handle:
            while positions < config.max_positions:
                if config.max_games is not None and games_seen >= config.max_games:
                    break
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games_seen += 1
                if not _passes_filters(game, filter_config):
                    continue

                board = game.board()
                used_this_game = False
                for ply, move in enumerate(game.mainline_moves()):
                    if positions >= config.max_positions:
                        break
                    if move not in board.legal_moves:
                        break
                    if ply % max(1, config.position_stride) == 0 and not board.is_game_over():
                        info = engine.analyse(board, limit)
                        pv = info.get("pv")
                        if pv:
                            best_move = pv[0]
                        else:
                            play = engine.play(board, limit)
                            best_move = play.move
                        if best_move in board.legal_moves:
                            score = info.get("score")
                            value = _score_to_value(score, board.turn)
                            boards.append(encode_board(board))
                            actions.append(move_to_action(best_move, board))
                            values.append(value)
                            fens.append(board.fen())
                            best_moves.append(best_move.uci())
                            positions += 1
                            used_this_game = True

                            if len(boards) >= config.chunk_size:
                                path = _write_teacher_chunk(
                                    out_dir,
                                    len(written),
                                    boards,
                                    actions,
                                    values,
                                    fens,
                                    best_moves,
                                    source=str(pgn_path),
                                )
                                written.append(path)
                                boards, actions, values, fens, best_moves = [], [], [], [], []
                    board.push(move)
                if used_this_game:
                    games_used += 1

    if boards:
        written.append(
            _write_teacher_chunk(
                out_dir,
                len(written),
                boards,
                actions,
                values,
                fens,
                best_moves,
                source=str(pgn_path),
            )
        )

    if not written:
        raise ValueError("No Stockfish teacher positions generated")

    (out_dir / "teacher_summary.txt").write_text(
        "\n".join(
            [
                f"source={pgn_path}",
                f"engine_path={config.engine_path}",
                f"games_seen={games_seen}",
                f"games_used={games_used}",
                f"positions={positions}",
                f"files={len(written)}",
                f"config={asdict(config)}",
            ]
        )
        + "\n"
    )
    return written


def _score_to_value(score: chess.engine.PovScore | None, turn: chess.Color) -> float:
    if score is None:
        return 0.0
    pov = score.pov(turn)
    centipawns = pov.score(mate_score=10000)
    if centipawns is None:
        return 0.0
    return float(np.tanh(centipawns / 600.0))


def _write_teacher_chunk(
    out_dir: Path,
    chunk_index: int,
    boards: list[np.ndarray],
    actions: list[int],
    values: list[float],
    fens: list[str],
    best_moves: list[str],
    source: str,
) -> Path:
    path = out_dir / f"teacher_{chunk_index:06d}.npz"
    np.savez_compressed(
        path,
        boards=np.asarray(boards, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        fens=np.asarray(fens),
        moves=np.asarray(best_moves),
        source=np.asarray(source),
    )
    return path
