from types import SimpleNamespace

import chess
import chess.engine
import chess.pgn

from alpha_chess.engine_self_play import EngineSelfPlayConfig, generate_engine_self_play


class FakeEngine:
    def __init__(self) -> None:
        self.quit_called = False

    def play(self, board: chess.Board, _limit: chess.engine.Limit) -> SimpleNamespace:
        return SimpleNamespace(move=next(iter(board.legal_moves), None))

    def analyse(
        self,
        board: chess.Board,
        _limit: chess.engine.Limit,
        multipv: int = 1,
    ) -> list[dict]:
        infos: list[dict] = []
        for index, move in enumerate(list(board.legal_moves)[:multipv]):
            infos.append(
                {
                    "pv": [move],
                    "score": chess.engine.PovScore(
                        chess.engine.Cp(100 - index * 10),
                        board.turn,
                    ),
                }
            )
        return infos

    def quit(self) -> None:
        self.quit_called = True


def test_engine_self_play_writes_legal_pgn_and_summary(tmp_path, monkeypatch) -> None:
    engines: list[FakeEngine] = []

    def fake_popen_uci(_path: str) -> FakeEngine:
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", fake_popen_uci)

    out = tmp_path / "engine_games.pgn"
    written = generate_engine_self_play(
        EngineSelfPlayConfig(
            out=str(out),
            engine_path="fake-stockfish",
            games=2,
            max_plies=4,
            opening_random_plies=2,
            opening_multipv=2,
            seed=7,
        )
    )

    assert written == out
    assert out.exists()
    assert engines and engines[0].quit_called
    summary = out.with_suffix(out.suffix + ".summary.txt").read_text()
    assert "games=2" in summary
    assert "average_plies=4.00" in summary

    games: list[chess.pgn.Game] = []
    with out.open(encoding="utf-8") as handle:
        while game := chess.pgn.read_game(handle):
            games.append(game)

    assert len(games) == 2
    for game in games:
        assert game.headers["Event"] == "AlphaChess engine self-play"
        assert game.headers["Result"] == "1/2-1/2"
        assert game.headers["PlyCount"] == "4"
        assert len(list(game.mainline_moves())) == 4


def test_engine_self_play_can_use_separate_engines(tmp_path, monkeypatch) -> None:
    opened_paths: list[str] = []
    engines: list[FakeEngine] = []

    def fake_popen_uci(path: str) -> FakeEngine:
        opened_paths.append(path)
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", fake_popen_uci)

    generate_engine_self_play(
        EngineSelfPlayConfig(
            out=str(tmp_path / "two_engines.pgn"),
            engine_path="unused",
            white_engine_path="white-engine",
            black_engine_path="black-engine",
            games=1,
            max_plies=2,
        )
    )

    assert opened_paths == ["white-engine", "black-engine"]
    assert all(engine.quit_called for engine in engines)
