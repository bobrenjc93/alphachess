"""Import expert PGN games into AlphaChess training NPZ files."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import zstandard

from alpha_chess.chess_env import ACTION_SIZE, encode_board, move_to_action


@dataclass
class PGNImportConfig:
    pgn: str
    out: str = "data/expert"
    max_games: int | None = None
    min_plies: int = 1
    chunk_size: int = 4096
    dense_policy: bool = False


def import_pgn(config: PGNImportConfig) -> list[Path]:
    pgn_path = Path(config.pgn)
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    boards: list[np.ndarray] = []
    policies: list[np.ndarray] | None = [] if config.dense_policy else None
    actions: list[int] = []
    values: list[float] = []
    fens: list[str] = []
    moves: list[str] = []
    games_seen = 0
    positions_seen = 0

    with _open_pgn_text(pgn_path) as handle:
        while config.max_games is None or games_seen < config.max_games:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games_seen += 1
            game_positions = _game_to_samples(game, dense_policy=config.dense_policy)
            if len(game_positions["boards"]) < config.min_plies:
                continue

            boards.extend(game_positions["boards"])
            actions.extend(game_positions["actions"])
            if policies is not None:
                policies.extend(game_positions["policies"])
            values.extend(game_positions["values"])
            fens.extend(game_positions["fens"])
            moves.extend(game_positions["moves"])

            while len(boards) >= config.chunk_size:
                path = _write_chunk(
                    out_dir,
                    len(written),
                    boards[: config.chunk_size],
                    actions[: config.chunk_size],
                    values[: config.chunk_size],
                    fens[: config.chunk_size],
                    moves[: config.chunk_size],
                    source=str(pgn_path),
                    policies=policies[: config.chunk_size] if policies is not None else None,
                )
                positions_seen += config.chunk_size
                written.append(path)
                del boards[: config.chunk_size]
                del actions[: config.chunk_size]
                if policies is not None:
                    del policies[: config.chunk_size]
                del values[: config.chunk_size]
                del fens[: config.chunk_size]
                del moves[: config.chunk_size]

    if boards:
        path = _write_chunk(
            out_dir,
            len(written),
            boards,
            actions,
            values,
            fens,
            moves,
            source=str(pgn_path),
            policies=policies,
        )
        positions_seen += len(boards)
        written.append(path)

    if not written:
        raise ValueError(f"No usable positions imported from {pgn_path}")

    summary = out_dir / "import_summary.txt"
    summary.write_text(
        f"source={pgn_path}\ngames_seen={games_seen}\npositions={positions_seen}\nfiles={len(written)}\n"
    )
    return written


def _open_pgn_text(path: Path) -> TextIO:
    """Open plain or compressed PGN as text.

    Supports common chess database archives: .pgn, .pgn.gz, .pgn.bz2,
    .pgn.xz, and .pgn.zst.
    """

    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".xz":
        return lzma.open(path, mode="rt", encoding="utf-8", errors="replace")
    if suffix == ".zst":
        compressed = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _game_to_samples(game: chess.pgn.Game, dense_policy: bool = False) -> dict[str, list]:
    result = game.headers.get("Result", "*")
    white_value = _result_to_white_value(result)
    board = game.board()

    boards: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    fens: list[str] = []
    moves: list[str] = []

    for move in game.mainline_moves():
        if move not in board.legal_moves:
            break
        action = move_to_action(move, board)
        boards.append(encode_board(board))
        actions.append(action)
        if dense_policy:
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[action] = 1.0
            policies.append(policy)
        values.append(white_value if board.turn == chess.WHITE else -white_value)
        fens.append(board.fen())
        moves.append(move.uci())
        board.push(move)

    return {
        "boards": boards,
        "policies": policies,
        "actions": actions,
        "values": values,
        "fens": fens,
        "moves": moves,
    }


def _result_to_white_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def _write_chunk(
    out_dir: Path,
    chunk_index: int,
    boards: list[np.ndarray],
    actions: list[int],
    values: list[float],
    fens: list[str],
    moves: list[str],
    source: str,
    policies: list[np.ndarray] | None = None,
) -> Path:
    path = out_dir / f"expert_{chunk_index:06d}.npz"
    payload = {
        "boards": np.asarray(boards, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float32),
        "fens": np.asarray(fens),
        "moves": np.asarray(moves),
        "source": np.asarray(source),
    }
    if policies is not None:
        payload["policies"] = np.asarray(policies, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path
