"""Generate Stockfish teacher labels for chess positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from alpha_chess.chess_env import ACTION_SIZE, encode_board, move_to_action
from alpha_chess.pgn_import import PGNImportConfig, _open_pgn_text, _passes_filters


@dataclass
class StockfishTeacherConfig:
    pgn: str | list[str]
    out: str = "data/teacher/stockfish"
    engine_path: str = "stockfish"
    engine_time: float = 0.02
    engine_depth: int | None = None
    max_games: int | None = None
    max_positions: int = 1024
    min_elo: int | None = None
    min_initial_seconds: int | None = None
    min_value_delta: float | None = None
    player_name: str | None = None
    multipv: int = 1
    policy_temperature_cp: float = 200.0
    position_stride: int = 4
    pv_plies: int = 0
    chunk_size: int = 1024


def generate_stockfish_teacher(config: StockfishTeacherConfig) -> list[Path]:
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_paths = _resolve_pgn_paths(config.pgn)
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)

    boards: list[np.ndarray] = []
    actions: list[int] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    value_deltas: list[float] = []
    fens: list[str] = []
    best_moves: list[str] = []
    written: list[Path] = []
    games_seen = 0
    games_used = 0
    positions = 0

    def flush_chunk(source: str) -> None:
        nonlocal boards, actions, policies, values, value_deltas, fens, best_moves
        if not boards:
            return
        written.append(
            _write_teacher_chunk(
                out_dir,
                len(written),
                boards,
                actions,
                values,
                value_deltas,
                fens,
                best_moves,
                source=source,
                policies=policies if policies else None,
            )
        )
        boards, actions, policies, values = [], [], [], []
        value_deltas, fens, best_moves = [], [], []

    def append_teacher_sample(
        source: str,
        sample_board: chess.Board,
        infos: list[dict],
        best_move: chess.Move,
        value: float,
        value_delta: float,
    ) -> None:
        nonlocal positions
        boards.append(encode_board(sample_board))
        actions.append(move_to_action(best_move, sample_board))
        if config.multipv > 1:
            policies.append(
                _policy_from_multipv(
                    sample_board,
                    infos,
                    config.policy_temperature_cp,
                    fallback_move=best_move,
                )
            )
        values.append(value)
        value_deltas.append(value_delta)
        fens.append(sample_board.fen())
        best_moves.append(best_move.uci())
        positions += 1
        if len(boards) >= config.chunk_size:
            flush_chunk(source)

    def append_pv_line_samples(
        source: str,
        root_board: chess.Board,
        root_infos: list[dict],
    ) -> int:
        added = 0
        line_board = root_board.copy(stack=False)
        pv = _best_pv_from_infos(root_infos)
        if not pv:
            return added
        next_move = pv[0]

        while added < max(0, config.pv_plies) and positions < config.max_positions:
            if next_move not in line_board.legal_moves:
                break
            line_board.push(next_move)
            if line_board.is_game_over(claim_draw=True):
                break

            infos = _analyse_position(engine, line_board, limit, config.multipv)
            best_move = _best_move_from_infos(infos)
            if best_move is None:
                play = engine.play(line_board, limit)
                best_move = play.move
            if best_move not in line_board.legal_moves:
                break

            score = infos[0].get("score") if infos else None
            value = _score_to_value(score, line_board.turn)
            append_teacher_sample(source, line_board, infos, best_move, value, 0.0)
            added += 1

            pv = _best_pv_from_infos(infos)
            if not pv:
                break
            next_move = pv[0]
        return added

    with chess.engine.SimpleEngine.popen_uci(config.engine_path) as engine:
        for pgn_path in pgn_paths:
            source = str(pgn_path)
            filter_config = PGNImportConfig(
                pgn=source,
                out=config.out,
                max_games=config.max_games,
                min_elo=config.min_elo,
                min_initial_seconds=config.min_initial_seconds,
            )
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
                        sample_position = ply % max(1, config.position_stride) == 0
                        sample_position = sample_position and not board.is_game_over()
                        if config.player_name is not None:
                            sample_position = sample_position and _matches_player_to_move(
                                game, board, config.player_name
                            )
                        if sample_position:
                            infos = _analyse_position(engine, board, limit, config.multipv)
                            best_move = _best_move_from_infos(infos)
                            if best_move is None:
                                play = engine.play(board, limit)
                                best_move = play.move
                            if best_move in board.legal_moves:
                                score = infos[0].get("score") if infos else None
                                value = _score_to_value(score, board.turn)
                                value_delta = 0.0
                                if config.min_value_delta is not None:
                                    after_board = board.copy(stack=False)
                                    after_board.push(move)
                                    after_info = engine.analyse(after_board, limit)
                                    value_delta = _value_drop_after_move(
                                        best_value=value,
                                        after_score=after_info.get("score"),
                                        after_turn=after_board.turn,
                                    )
                                    if value_delta < config.min_value_delta:
                                        board.push(move)
                                        continue
                                append_teacher_sample(
                                    source,
                                    board,
                                    infos,
                                    best_move,
                                    value,
                                    value_delta,
                                )
                                if positions < config.max_positions:
                                    append_pv_line_samples(source, board, infos)
                                used_this_game = True
                        board.push(move)
                    if used_this_game:
                        games_used += 1

            flush_chunk(source)
            if positions >= config.max_positions:
                break

    if not written:
        raise ValueError("No Stockfish teacher positions generated")

    (out_dir / "teacher_summary.txt").write_text(
        "\n".join(
            [
                f"source={pgn_paths[0] if len(pgn_paths) == 1 else 'multiple'}",
                f"sources={[str(path) for path in pgn_paths]}",
                f"engine_path={config.engine_path}",
                f"games_seen={games_seen}",
                f"games_used={games_used}",
                f"positions={positions}",
                f"files={len(written)}",
                f"min_value_delta={config.min_value_delta}",
                f"player_name={config.player_name}",
                f"multipv={config.multipv}",
                f"policy_temperature_cp={config.policy_temperature_cp}",
                f"pv_plies={config.pv_plies}",
                f"config={asdict(config)}",
            ]
        )
        + "\n"
    )
    return written


def _resolve_pgn_paths(pgn: str | list[str]) -> list[Path]:
    if isinstance(pgn, str):
        return [Path(pgn)]
    return [Path(path) for path in pgn]


def _analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    multipv: int,
) -> list[dict]:
    if multipv > 1:
        analysis = engine.analyse(board, limit, multipv=max(1, multipv))
    else:
        analysis = engine.analyse(board, limit)
    return analysis if isinstance(analysis, list) else [analysis]


def _best_move_from_infos(infos: list[dict]) -> chess.Move | None:
    for info in infos:
        pv = info.get("pv")
        if pv:
            return pv[0]
    return None


def _best_pv_from_infos(infos: list[dict]) -> list[chess.Move]:
    for info in infos:
        pv = info.get("pv")
        if pv:
            return list(pv)
    return []


def _score_to_value(score: chess.engine.PovScore | None, turn: chess.Color) -> float:
    if score is None:
        return 0.0
    pov = score.pov(turn)
    centipawns = pov.score(mate_score=10000)
    if centipawns is None:
        return 0.0
    return float(np.tanh(centipawns / 600.0))


def _matches_player_to_move(game: chess.pgn.Game, board: chess.Board, player_name: str) -> bool:
    header = "White" if board.turn == chess.WHITE else "Black"
    return game.headers.get(header, "").strip().casefold() == player_name.strip().casefold()


def _policy_from_multipv(
    board: chess.Board,
    infos: list[dict],
    temperature_cp: float,
    fallback_move: chess.Move | None = None,
) -> np.ndarray:
    action_scores: dict[int, float] = {}
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        centipawns = score.pov(board.turn).score(mate_score=10000)
        if centipawns is None:
            continue
        action = move_to_action(move, board)
        action_scores[action] = max(float(centipawns), action_scores.get(action, -float("inf")))

    if not action_scores and fallback_move is not None and fallback_move in board.legal_moves:
        action_scores[move_to_action(fallback_move, board)] = 0.0

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    if not action_scores:
        return policy

    actions = np.asarray(list(action_scores.keys()), dtype=np.int64)
    scores = np.asarray([action_scores[int(action)] for action in actions], dtype=np.float64)
    if temperature_cp <= 0:
        policy[int(actions[int(np.argmax(scores))])] = 1.0
        return policy

    logits = scores / temperature_cp
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    total = float(probabilities.sum())
    if total <= 0 or not np.isfinite(total):
        policy[int(actions[int(np.argmax(scores))])] = 1.0
    else:
        policy[actions] = (probabilities / total).astype(np.float32)
    return policy


def _value_drop_after_move(
    best_value: float,
    after_score: chess.engine.PovScore | None,
    after_turn: chess.Color,
) -> float:
    # after_score is from the opponent-to-move side after the game move.
    # Negating maps it back to the original side-to-move perspective.
    original_value_after_game_move = -_score_to_value(after_score, after_turn)
    return best_value - original_value_after_game_move


def _write_teacher_chunk(
    out_dir: Path,
    chunk_index: int,
    boards: list[np.ndarray],
    actions: list[int],
    values: list[float],
    value_deltas: list[float],
    fens: list[str],
    best_moves: list[str],
    source: str,
    policies: list[np.ndarray] | None = None,
) -> Path:
    path = out_dir / f"teacher_{chunk_index:06d}.npz"
    payload = {
        "boards": np.asarray(boards, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float32),
        "value_deltas": np.asarray(value_deltas, dtype=np.float32),
        "fens": np.asarray(fens),
        "moves": np.asarray(best_moves),
        "source": np.asarray(source),
    }
    if policies is not None:
        payload["policies"] = np.asarray(policies, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path
