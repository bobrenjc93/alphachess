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
    player_score_min: float | None = None
    player_score_max: float | None = None
    multipv: int = 1
    policy_temperature_cp: float = 200.0
    position_stride: int = 4
    skip_positions: int = 0
    min_ply: int = 0
    max_ply: int | None = None
    pv_plies: int = 0
    game_line_plies: int = 0
    blunder_context_plies: int = 0
    first_blunder_only: bool = False
    legal_bad_actions_per_position: int = 0
    legal_bad_action_min_delta: float | None = None
    chunk_size: int = 1024


def generate_stockfish_teacher(config: StockfishTeacherConfig) -> list[Path]:
    if config.blunder_context_plies > 0 and config.min_value_delta is None:
        raise ValueError("blunder_context_plies requires min_value_delta")
    if (
        config.legal_bad_actions_per_position > 0
        and config.legal_bad_action_min_delta is None
        and config.min_value_delta is None
    ):
        raise ValueError(
            "legal_bad_actions_per_position requires legal_bad_action_min_delta "
            "or min_value_delta"
        )
    if (config.player_score_min is not None or config.player_score_max is not None) and (
        config.player_name is None
    ):
        raise ValueError("player_score_min/player_score_max require player_name")

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
    bad_actions: list[list[int]] = []
    bad_action_deltas: list[list[float]] = []
    played_moves: list[str] = []
    written: list[Path] = []
    games_seen = 0
    games_used = 0
    positions = 0
    skipped_positions = 0
    legal_bad_action_rows = 0
    legal_bad_action_labels = 0
    legal_bad_action_candidates_evaluated = 0

    def flush_chunk(source: str) -> None:
        nonlocal boards, actions, policies, values, value_deltas, fens, best_moves
        nonlocal bad_actions, bad_action_deltas, played_moves
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
                bad_actions,
                bad_action_deltas,
                played_moves,
                source=source,
                policies=policies if policies else None,
            )
        )
        boards, actions, policies, values = [], [], [], []
        value_deltas, fens, best_moves = [], [], []
        bad_actions, bad_action_deltas, played_moves = [], [], []

    def append_teacher_sample(
        source: str,
        sample_board: chess.Board,
        infos: list[dict],
        best_move: chess.Move,
        value: float,
        value_delta: float,
        bad_move: chess.Move | None = None,
        legal_bad_moves: list[tuple[chess.Move, float]] | None = None,
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
        bad_action_row: list[int] = []
        bad_delta_row: list[float] = []
        if bad_move is not None and bad_move in sample_board.legal_moves and bad_move != best_move:
            bad_action_row.append(move_to_action(bad_move, sample_board))
            bad_delta_row.append(float(value_delta))
            played_moves.append(bad_move.uci())
        else:
            played_moves.append("")
        max_bad_actions = (
            max(1, int(config.legal_bad_actions_per_position))
            if config.legal_bad_actions_per_position > 0
            else None
        )
        for move, delta in legal_bad_moves or []:
            if max_bad_actions is not None and len(bad_action_row) >= max_bad_actions:
                break
            if move not in sample_board.legal_moves or move == best_move:
                continue
            action = move_to_action(move, sample_board)
            if action in bad_action_row:
                continue
            bad_action_row.append(action)
            bad_delta_row.append(float(delta))
        if not bad_action_row:
            bad_action_row.append(-1)
            bad_delta_row.append(0.0)
        bad_actions.append(bad_action_row)
        bad_action_deltas.append(bad_delta_row)
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

    def append_game_line_samples(
        source: str,
        root_board: chess.Board,
        continuation: list[chess.Move],
    ) -> int:
        added = 0
        line_board = root_board.copy(stack=False)
        for next_move in continuation:
            if added >= max(0, config.game_line_plies) or positions >= config.max_positions:
                break
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
        return added

    def append_blunder_context_samples(
        source: str,
        context_boards: list[chess.Board],
    ) -> int:
        added = 0
        context_plies = max(0, config.blunder_context_plies)
        if context_plies <= 0:
            return added

        for context_board in context_boards[-context_plies:]:
            if positions >= config.max_positions:
                break
            if context_board.is_game_over(claim_draw=True):
                continue

            infos = _analyse_position(engine, context_board, limit, config.multipv)
            best_move = _best_move_from_infos(infos)
            if best_move is None:
                play = engine.play(context_board, limit)
                best_move = play.move
            if best_move not in context_board.legal_moves:
                continue

            score = infos[0].get("score") if infos else None
            value = _score_to_value(score, context_board.turn)
            append_teacher_sample(source, context_board, infos, best_move, value, 0.0)
            added += 1
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
                    if not _passes_player_score_filter(
                        game,
                        config.player_name,
                        config.player_score_min,
                        config.player_score_max,
                    ):
                        continue

                    board = game.board()
                    mainline_moves = list(game.mainline_moves())
                    context_boards: list[chess.Board] = []
                    found_blunder_this_game = False
                    used_this_game = False
                    for ply, move in enumerate(mainline_moves):
                        if positions >= config.max_positions:
                            break
                        if config.first_blunder_only and found_blunder_this_game:
                            break
                        if move not in board.legal_moves:
                            break
                        sample_position = ply % max(1, config.position_stride) == 0
                        sample_position = sample_position and not board.is_game_over()
                        sample_position = sample_position and ply >= max(0, config.min_ply)
                        if config.max_ply is not None:
                            sample_position = sample_position and ply <= config.max_ply
                        if config.player_name is not None:
                            sample_position = sample_position and _matches_player_to_move(
                                game, board, config.player_name
                            )
                        if sample_position:
                            if skipped_positions < max(0, config.skip_positions):
                                skipped_positions += 1
                                board.push(move)
                                continue
                            context_board = board.copy(stack=False)
                            infos = _analyse_position(engine, board, limit, config.multipv)
                            best_move = _best_move_from_infos(infos)
                            if best_move is None:
                                play = engine.play(board, limit)
                                best_move = play.move
                            if best_move in board.legal_moves:
                                score = infos[0].get("score") if infos else None
                                value = _score_to_value(score, board.turn)
                                value_delta = 0.0
                                should_append = True
                                legal_bad_moves: list[tuple[chess.Move, float]] = []
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
                                        should_append = False
                                bad_move = (
                                    move
                                    if config.min_value_delta is not None
                                    and value_delta > 0.0
                                    else None
                                )
                                if config.legal_bad_actions_per_position > 0:
                                    (
                                        legal_bad_moves,
                                        candidates_evaluated,
                                    ) = _legal_bad_moves(
                                        engine=engine,
                                        board=board,
                                        best_move=best_move,
                                        best_value=value,
                                        limit=limit,
                                        min_value_delta=(
                                            config.legal_bad_action_min_delta
                                            if config.legal_bad_action_min_delta is not None
                                            else float(config.min_value_delta)
                                        ),
                                        max_bad_actions=config.legal_bad_actions_per_position,
                                    )
                                    legal_bad_action_candidates_evaluated += candidates_evaluated
                                    if legal_bad_moves:
                                        should_append = True
                                        legal_bad_action_rows += 1
                                        legal_bad_action_labels += len(legal_bad_moves)
                                    elif config.min_value_delta is None:
                                        should_append = False
                                if should_append:
                                    append_teacher_sample(
                                        source,
                                        board,
                                        infos,
                                        best_move,
                                        value,
                                        value_delta,
                                        bad_move=bad_move,
                                        legal_bad_moves=legal_bad_moves,
                                    )
                                    if bad_move is not None and positions < config.max_positions:
                                        append_blunder_context_samples(source, context_boards)
                                    if positions < config.max_positions:
                                        append_pv_line_samples(source, board, infos)
                                    if positions < config.max_positions:
                                        append_game_line_samples(
                                            source,
                                            board,
                                            mainline_moves[ply:],
                                        )
                                    used_this_game = True
                                    if bad_move is not None:
                                        found_blunder_this_game = True
                            context_boards.append(context_board)
                            context_plies = max(0, config.blunder_context_plies)
                            if context_plies > 0 and len(context_boards) > context_plies:
                                context_boards = context_boards[-context_plies:]
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
                f"skip_positions={config.skip_positions}",
                f"skipped_positions={skipped_positions}",
                f"positions={positions}",
                f"files={len(written)}",
                f"min_value_delta={config.min_value_delta}",
                f"player_name={config.player_name}",
                f"player_score_min={config.player_score_min}",
                f"player_score_max={config.player_score_max}",
                f"multipv={config.multipv}",
                f"policy_temperature_cp={config.policy_temperature_cp}",
                f"min_ply={config.min_ply}",
                f"max_ply={config.max_ply}",
                f"pv_plies={config.pv_plies}",
                f"game_line_plies={config.game_line_plies}",
                f"blunder_context_plies={config.blunder_context_plies}",
                f"first_blunder_only={config.first_blunder_only}",
                f"legal_bad_actions_per_position={config.legal_bad_actions_per_position}",
                f"legal_bad_action_min_delta={config.legal_bad_action_min_delta}",
                f"legal_bad_action_rows={legal_bad_action_rows}",
                f"legal_bad_action_labels={legal_bad_action_labels}",
                f"legal_bad_action_candidates_evaluated={legal_bad_action_candidates_evaluated}",
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


def _player_score(game: chess.pgn.Game, player_name: str) -> float | None:
    score_header = f"{player_name}Score"
    if score_header in game.headers:
        try:
            return float(game.headers[score_header])
        except ValueError:
            return None

    normalized_score_header = score_header.casefold()
    for key, value in game.headers.items():
        if key.casefold() != normalized_score_header:
            continue
        try:
            return float(value)
        except ValueError:
            return None

    result = game.headers.get("Result", "*")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None

    player = player_name.strip().casefold()
    white = game.headers.get("White", "").strip().casefold()
    black = game.headers.get("Black", "").strip().casefold()
    if white == player:
        return {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[result]
    if black == player:
        return {"1-0": 0.0, "1/2-1/2": 0.5, "0-1": 1.0}[result]
    return None


def _passes_player_score_filter(
    game: chess.pgn.Game,
    player_name: str | None,
    score_min: float | None,
    score_max: float | None,
) -> bool:
    if score_min is None and score_max is None:
        return True
    if player_name is None:
        return False
    score = _player_score(game, player_name)
    if score is None:
        return False
    if score_min is not None and score < score_min:
        return False
    if score_max is not None and score > score_max:
        return False
    return True


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


def _legal_bad_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    best_move: chess.Move,
    best_value: float,
    limit: chess.engine.Limit,
    min_value_delta: float,
    max_bad_actions: int,
) -> tuple[list[tuple[chess.Move, float]], int]:
    bad_moves: list[tuple[chess.Move, float]] = []
    candidates_evaluated = 0
    for candidate in board.legal_moves:
        if candidate == best_move:
            continue
        candidates_evaluated += 1
        after_board = board.copy(stack=False)
        after_board.push(candidate)
        after_info = engine.analyse(after_board, limit)
        value_delta = _value_drop_after_move(
            best_value=best_value,
            after_score=after_info.get("score"),
            after_turn=after_board.turn,
        )
        if value_delta < min_value_delta:
            continue
        bad_moves.append((candidate, float(value_delta)))

    bad_moves.sort(key=lambda item: item[1], reverse=True)
    return bad_moves[: max(1, int(max_bad_actions))], candidates_evaluated


def _pad_bad_action_rows(rows: list[list[int]], fill: int = -1) -> np.ndarray:
    if not rows:
        return np.empty((0,), dtype=np.int64)
    max_width = max(len(row) for row in rows)
    if max_width <= 1:
        return np.asarray([row[0] if row else fill for row in rows], dtype=np.int64)
    padded = np.full((len(rows), max_width), fill, dtype=np.int64)
    for row_index, row in enumerate(rows):
        if row:
            padded[row_index, : len(row)] = row
    return padded


def _pad_bad_action_delta_rows(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.empty((0,), dtype=np.float32)
    max_width = max(len(row) for row in rows)
    if max_width <= 1:
        return np.asarray([row[0] if row else 0.0 for row in rows], dtype=np.float32)
    padded = np.zeros((len(rows), max_width), dtype=np.float32)
    for row_index, row in enumerate(rows):
        if row:
            padded[row_index, : len(row)] = row
    return padded


def _write_teacher_chunk(
    out_dir: Path,
    chunk_index: int,
    boards: list[np.ndarray],
    actions: list[int],
    values: list[float],
    value_deltas: list[float],
    fens: list[str],
    best_moves: list[str],
    bad_actions: list[list[int]],
    bad_action_deltas: list[list[float]],
    played_moves: list[str],
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
        "bad_actions": _pad_bad_action_rows(bad_actions),
        "bad_action_deltas": _pad_bad_action_delta_rows(bad_action_deltas),
        "played_moves": np.asarray(played_moves),
        "source": np.asarray(source),
    }
    if policies is not None:
        payload["policies"] = np.asarray(policies, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path
