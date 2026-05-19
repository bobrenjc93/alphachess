"""Import Lichess puzzle CSV data as tactical training positions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from alpha_chess.chess_env import encode_board, move_to_action
from alpha_chess.pgn_import import _open_pgn_text


@dataclass
class PuzzleImportConfig:
    puzzles: str
    out: str = "data/puzzles"
    max_positions: int | None = None
    min_rating: int | None = None
    max_rating: int | None = None
    theme: str | None = None
    chunk_size: int = 4096
    value: float = 1.0
    include_solution_line: bool = False


def import_puzzles(config: PuzzleImportConfig) -> list[Path]:
    puzzle_path = Path(config.puzzles)
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    boards: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    fens: list[str] = []
    moves: list[str] = []
    solutions: list[str] = []
    ratings: list[int] = []
    themes: list[str] = []
    written: list[Path] = []
    rows_seen = 0
    rows_imported = 0
    rows_skipped = 0
    positions = 0

    with _open_pgn_text(puzzle_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if config.max_positions is not None and positions >= config.max_positions:
                break
            rows_seen += 1
            if not _passes_puzzle_filters(row, config):
                rows_skipped += 1
                continue

            try:
                board = chess.Board(row["FEN"])
                solution_moves = row["Moves"].split()
                first_move = chess.Move.from_uci(solution_moves[0])
            except (KeyError, IndexError, ValueError):
                rows_skipped += 1
                continue
            if first_move not in board.legal_moves:
                rows_skipped += 1
                continue

            winning_color = board.turn
            imported_from_row = 0
            moves_to_import = solution_moves if config.include_solution_line else solution_moves[:1]
            for move_text in moves_to_import:
                if config.max_positions is not None and positions >= config.max_positions:
                    break
                try:
                    move = chess.Move.from_uci(move_text)
                except ValueError:
                    break
                if move not in board.legal_moves:
                    break

                boards.append(encode_board(board))
                actions.append(move_to_action(move, board))
                values.append(config.value if board.turn == winning_color else -config.value)
                fens.append(board.fen())
                moves.append(move.uci())
                solutions.append(row.get("Moves", ""))
                ratings.append(_safe_int(row.get("Rating")) or 0)
                themes.append(row.get("Themes", ""))
                imported_from_row += 1
                positions += 1
                board.push(move)

                if len(boards) >= config.chunk_size:
                    written.append(
                        _write_puzzle_chunk(
                            out_dir,
                            len(written),
                            boards,
                            actions,
                            values,
                            fens,
                            moves,
                            solutions,
                            ratings,
                            themes,
                            source=str(puzzle_path),
                        )
                    )
                    boards, actions, values, fens = [], [], [], []
                    moves, solutions, ratings, themes = [], [], [], []

            if imported_from_row == 0:
                rows_skipped += 1
                continue
            rows_imported += 1


    if boards:
        written.append(
            _write_puzzle_chunk(
                out_dir,
                len(written),
                boards,
                actions,
                values,
                fens,
                moves,
                solutions,
                ratings,
                themes,
                source=str(puzzle_path),
            )
        )

    if not written:
        raise ValueError(f"No usable puzzle positions imported from {puzzle_path}")

    (out_dir / "puzzle_summary.txt").write_text(
        "\n".join(
            [
                f"source={puzzle_path}",
                f"rows_seen={rows_seen}",
                f"rows_imported={rows_imported}",
                f"rows_skipped={rows_skipped}",
                f"positions={positions}",
                f"files={len(written)}",
                f"config={asdict(config)}",
            ]
        )
        + "\n"
    )
    return written


def _passes_puzzle_filters(row: dict[str, str], config: PuzzleImportConfig) -> bool:
    rating = _safe_int(row.get("Rating"))
    if config.min_rating is not None and (rating is None or rating < config.min_rating):
        return False
    if config.max_rating is not None and (rating is None or rating > config.max_rating):
        return False
    if config.theme is not None:
        theme_set = set(row.get("Themes", "").split())
        if config.theme not in theme_set:
            return False
    return True


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _write_puzzle_chunk(
    out_dir: Path,
    chunk_index: int,
    boards: list[np.ndarray],
    actions: list[int],
    values: list[float],
    fens: list[str],
    moves: list[str],
    solutions: list[str],
    ratings: list[int],
    themes: list[str],
    source: str,
) -> Path:
    path = out_dir / f"puzzles_{chunk_index:06d}.npz"
    np.savez_compressed(
        path,
        boards=np.asarray(boards, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        fens=np.asarray(fens),
        moves=np.asarray(moves),
        solutions=np.asarray(solutions),
        ratings=np.asarray(ratings, dtype=np.int32),
        themes=np.asarray(themes),
        source=np.asarray(source),
    )
    return path
