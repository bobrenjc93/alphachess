from types import SimpleNamespace

import chess
import chess.engine
import chess.pgn
import numpy as np

from alpha_chess.chess_env import move_to_action
from alpha_chess.stockfish_teacher import (
    StockfishTeacherConfig,
    _matches_player_to_move,
    _passes_player_score_filter,
    _policy_from_multipv,
    _player_score,
    _resolve_pgn_paths,
    _score_to_value,
    _value_drop_after_move,
    generate_stockfish_teacher,
)


def test_score_to_value_from_side_to_move() -> None:
    score = chess.engine.PovScore(chess.engine.Cp(300), chess.WHITE)
    assert _score_to_value(score, chess.WHITE) > 0
    assert _score_to_value(score, chess.BLACK) < 0


def test_value_drop_after_move_maps_back_to_original_side() -> None:
    best_value = 0.5
    after_score = chess.engine.PovScore(chess.engine.Cp(120), chess.BLACK)
    assert _value_drop_after_move(best_value, after_score, chess.BLACK) > 0.6


def test_matches_player_to_move_uses_pgn_headers() -> None:
    game = chess.pgn.Game()
    game.headers["White"] = "AlphaChess"
    game.headers["Black"] = "Stockfish"
    board = game.board()

    assert _matches_player_to_move(game, board, "alphachess")

    board.push(chess.Move.from_uci("e2e4"))
    assert not _matches_player_to_move(game, board, "AlphaChess")
    assert _matches_player_to_move(game, board, "Stockfish")


def test_player_score_prefers_explicit_eval_header() -> None:
    game = chess.pgn.Game()
    game.headers["White"] = "AlphaChess"
    game.headers["Black"] = "Stockfish"
    game.headers["Result"] = "1-0"
    game.headers["AlphaChessScore"] = "0.0"

    assert _player_score(game, "AlphaChess") == 0.0
    assert _passes_player_score_filter(game, "AlphaChess", None, 0.0)
    assert not _passes_player_score_filter(game, "AlphaChess", 0.5, None)


def test_player_score_falls_back_to_result_and_color() -> None:
    game = chess.pgn.Game()
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "AlphaChess"
    game.headers["Result"] = "0-1"

    assert _player_score(game, "AlphaChess") == 1.0
    assert _player_score(game, "Stockfish") == 0.0


def test_resolve_pgn_paths_accepts_one_or_many() -> None:
    assert [str(path) for path in _resolve_pgn_paths("one.pgn")] == ["one.pgn"]
    assert [str(path) for path in _resolve_pgn_paths(["one.pgn", "two.pgn"])] == [
        "one.pgn",
        "two.pgn",
    ]


def test_policy_from_multipv_builds_soft_legal_distribution() -> None:
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    policy = _policy_from_multipv(
        board,
        [
            {"pv": [e4], "score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE)},
            {"pv": [d4], "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)},
        ],
        temperature_cp=100.0,
    )

    e4_action = move_to_action(e4, board)
    d4_action = move_to_action(d4, board)
    assert abs(float(policy.sum()) - 1.0) < 1e-6
    assert policy[e4_action] > policy[d4_action] > 0


def test_stockfish_teacher_can_include_pv_line(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            move = self._best_move(board)
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=self._best_move(board))

        def _best_move(self, board):
            for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    return move
            return next(iter(board.legal_moves))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher"),
            engine_path="fake-stockfish",
            max_positions=3,
            player_name="AlphaChess",
            position_stride=99,
            pv_plies=2,
        )
    )

    data = np.load(paths[0])

    assert data["moves"].tolist() == ["e2e4", "e7e5", "g1f3"]
    assert data["boards"].shape[0] == 3
    assert "pv_plies=2" in (tmp_path / "teacher" / "teacher_summary.txt").read_text()


def test_stockfish_teacher_can_include_game_line(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. e4 d5 2. exd5 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-game-line"),
            engine_path="fake-stockfish",
            max_positions=3,
            position_stride=99,
            game_line_plies=2,
        )
    )

    data = np.load(paths[0])
    boards = [chess.Board(str(fen)) for fen in data["fens"]]

    assert data["fens"].shape[0] == 3
    assert boards[0].ply() == 0
    assert boards[1].ply() == 1
    assert boards[1].piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert boards[2].ply() == 2
    assert boards[2].piece_at(chess.D5) == chess.Piece(chess.PAWN, chess.BLACK)
    assert "game_line_plies=2" in (
        tmp_path / "teacher-game-line" / "teacher_summary.txt"
    ).read_text()


def test_stockfish_teacher_skips_root_samples_before_analysis(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __init__(self) -> None:
            self.analyse_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            self.analyse_calls += 1
            move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    engine = FakeEngine()
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: engine)

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-skip"),
            engine_path="fake-stockfish",
            max_positions=2,
            position_stride=1,
            skip_positions=2,
        )
    )

    data = np.load(paths[0])
    boards = [chess.Board(str(fen)) for fen in data["fens"]]
    summary = (tmp_path / "teacher-skip" / "teacher_summary.txt").read_text()

    assert [board.ply() for board in boards] == [2, 3]
    assert engine.analyse_calls == 2
    assert "skip_positions=2" in summary
    assert "skipped_positions=2" in summary


def test_stockfish_teacher_stores_played_bad_action(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. d4 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            if board.ply() == 0:
                move = chess.Move.from_uci("e2e4")
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            else:
                move = next(iter(board.legal_moves))
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            info = {"pv": [move], "score": score}
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-bad-action"),
            engine_path="fake-stockfish",
            max_positions=1,
            min_value_delta=0.1,
            player_name="AlphaChess",
            position_stride=1,
        )
    )

    data = np.load(paths[0])
    board = chess.Board()

    assert data["moves"].tolist() == ["e2e4"]
    assert data["played_moves"].tolist() == ["d2d4"]
    assert int(data["bad_actions"][0]) == move_to_action(chess.Move.from_uci("d2d4"), board)


def test_stockfish_teacher_can_store_multiple_legal_bad_actions(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. d4 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            if not board.move_stack:
                move = chess.Move.from_uci("e2e4")
                info = {
                    "pv": [move],
                    "score": chess.engine.PovScore(chess.engine.Cp(300), board.turn),
                }
                return [info] if multipv and multipv > 1 else info

            last_move = board.peek()
            cp = 300 if last_move.uci() in {"d2d4", "g1f3"} else -300
            return {
                "pv": [next(iter(board.legal_moves))],
                "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
            }

        def play(self, board, _limit):
            return SimpleNamespace(move=chess.Move.from_uci("e2e4"))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-legal-bad"),
            engine_path="fake-stockfish",
            max_positions=1,
            min_value_delta=0.2,
            legal_bad_actions_per_position=2,
            legal_value_policy_temperature=0.2,
        )
    )

    data = np.load(paths[0])
    board = chess.Board()
    e4_action = move_to_action(chess.Move.from_uci("e2e4"), board)
    d4_action = move_to_action(chess.Move.from_uci("d2d4"), board)
    nf3_action = move_to_action(chess.Move.from_uci("g1f3"), board)

    assert data["bad_actions"].shape == (1, 2)
    assert set(data["bad_actions"][0].tolist()) == {d4_action, nf3_action}
    assert data["bad_action_deltas"].shape == (1, 2)
    assert abs(float(data["policies"][0].sum()) - 1.0) < 1e-6
    assert data["policies"][0, e4_action] > data["policies"][0, d4_action]
    assert data["policies"][0, e4_action] > data["policies"][0, nf3_action]
    assert "legal_bad_action_labels=2" in (
        tmp_path / "teacher-legal-bad" / "teacher_summary.txt"
    ).read_text()


def test_stockfish_teacher_can_filter_by_player_score(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "1-0"]',
                '[AlphaChessScore "1.0"]',
                "",
                "1. d4 *",
                "",
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "0-1"]',
                '[AlphaChessScore "0.0"]',
                "",
                "1. d4 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            if board.ply() == 0:
                move = chess.Move.from_uci("e2e4")
            else:
                move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(300), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-score-filter"),
            engine_path="fake-stockfish",
            max_positions=10,
            min_value_delta=0.1,
            player_name="AlphaChess",
            player_score_max=0.0,
            position_stride=1,
        )
    )

    data = np.load(paths[0])
    summary = (tmp_path / "teacher-score-filter" / "teacher_summary.txt").read_text()

    assert data["fens"].shape[0] == 1
    assert data["moves"].tolist() == ["e2e4"]
    assert "games_seen=2" in summary
    assert "games_used=1" in summary
    assert "player_score_max=0.0" in summary


def test_stockfish_teacher_can_backfill_blunder_context(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. e4 e5 2. d4 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            if board.ply() == 0:
                move = chess.Move.from_uci("e2e4")
                score = chess.engine.PovScore(chess.engine.Cp(0), board.turn)
            elif board.ply() == 1:
                move = next(iter(board.legal_moves))
                score = chess.engine.PovScore(chess.engine.Cp(0), board.turn)
            elif board.turn == chess.WHITE:
                move = chess.Move.from_uci("d2d3")
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            else:
                move = next(iter(board.legal_moves))
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            info = {"pv": [move], "score": score}
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-context"),
            engine_path="fake-stockfish",
            max_positions=2,
            min_value_delta=0.1,
            player_name="AlphaChess",
            position_stride=1,
            blunder_context_plies=1,
        )
    )

    data = np.load(paths[0])
    initial_board = chess.Board()
    blunder_board = chess.Board()
    blunder_board.push(chess.Move.from_uci("e2e4"))
    blunder_board.push(chess.Move.from_uci("e7e5"))

    assert data["moves"].tolist() == ["d2d3", "e2e4"]
    assert int(data["bad_actions"][0]) == move_to_action(
        chess.Move.from_uci("d2d4"),
        blunder_board,
    )
    assert int(data["bad_actions"][1]) == -1
    assert data["fens"].tolist() == [blunder_board.fen(), initial_board.fen()]
    assert "blunder_context_plies=1" in (
        tmp_path / "teacher-context" / "teacher_summary.txt"
    ).read_text()


def test_stockfish_teacher_can_stop_after_first_blunder(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. d4 d5 2. c4 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            if board.turn == chess.WHITE:
                move = chess.Move.from_uci("e2e4") if board.ply() == 0 else next(iter(board.legal_moves))
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            else:
                move = next(iter(board.legal_moves))
                score = chess.engine.PovScore(chess.engine.Cp(300), board.turn)
            info = {"pv": [move], "score": score}
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-first-blunder"),
            engine_path="fake-stockfish",
            max_positions=10,
            min_value_delta=0.1,
            player_name="AlphaChess",
            position_stride=1,
            first_blunder_only=True,
        )
    )

    data = np.load(paths[0])
    board = chess.Board()

    assert data["moves"].tolist() == ["e2e4"]
    assert int(data["bad_actions"][0]) == move_to_action(chess.Move.from_uci("d2d4"), board)
    assert data["fens"].shape[0] == 1
    assert "first_blunder_only=True" in (
        tmp_path / "teacher-first-blunder" / "teacher_summary.txt"
    ).read_text()


def test_stockfish_teacher_respects_ply_window(monkeypatch, tmp_path) -> None:
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        "\n".join(
            [
                '[Event "?"]',
                '[White "AlphaChess"]',
                '[Black "Stockfish"]',
                '[Result "*"]',
                "",
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *",
                "",
            ]
        )
    )

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            pgn=str(pgn_path),
            out=str(tmp_path / "teacher-window"),
            engine_path="fake-stockfish",
            max_positions=10,
            position_stride=1,
            min_ply=2,
            max_ply=3,
        )
    )

    data = np.load(paths[0])

    assert data["fens"].shape[0] == 2
    assert all(chess.Board(str(fen)).fullmove_number == 2 for fen in data["fens"])
    summary = (tmp_path / "teacher-window" / "teacher_summary.txt").read_text()
    assert "min_ply=2" in summary
    assert "max_ply=3" in summary


def test_stockfish_teacher_can_generate_from_fen_file(monkeypatch, tmp_path) -> None:
    fen_path = tmp_path / "positions.fen"
    fen_rows = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    fen_path.write_text("\n".join(["# target positions", *fen_rows, ""]))

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            fen_file=str(fen_path),
            out=str(tmp_path / "teacher-fens"),
            engine_path="fake-stockfish",
            max_positions=10,
            multipv=2,
        )
    )

    data = np.load(paths[0])

    assert data["fens"].tolist() == fen_rows
    assert data["actions"].shape[0] == 2
    assert data["policies"].shape[0] == 2
    summary = (tmp_path / "teacher-fens" / "teacher_summary.txt").read_text()
    assert f"fen_sources={[str(fen_path)]}" in summary
    assert "fen_positions_seen=2" in summary
    assert "fen_positions_used=2" in summary


def test_stockfish_teacher_can_branch_from_fen_file(monkeypatch, tmp_path) -> None:
    fen_path = tmp_path / "positions.fen"
    fen_path.write_text(chess.STARTING_FEN + "\n")

    class FakeEngine:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def analyse(self, board, _limit, multipv=None):
            move = next(iter(board.legal_moves))
            info = {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
            return [info] if multipv and multipv > 1 else info

        def play(self, board, _limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: FakeEngine())

    paths = generate_stockfish_teacher(
        StockfishTeacherConfig(
            fen_file=str(fen_path),
            out=str(tmp_path / "teacher-branch-fens"),
            engine_path="fake-stockfish",
            max_positions=10,
            multipv=2,
            fen_branch_plies=1,
            fen_branch_width=1,
        )
    )

    data = np.load(paths[0])
    boards = [chess.Board(str(fen)) for fen in data["fens"]]

    assert len(boards) == 2
    assert boards[0].fen() == chess.STARTING_FEN
    assert boards[1].turn == chess.BLACK
    summary = (tmp_path / "teacher-branch-fens" / "teacher_summary.txt").read_text()
    assert "fen_branch_plies=1" in summary
    assert "fen_branch_width=1" in summary
    assert "fen_positions_seen=2" in summary
    assert "fen_positions_used=2" in summary
