import textwrap

import numpy as np
import zstandard

from alpha_chess.chess_env import ACTION_SIZE, NUM_INPUT_PLANES
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.pgn_import import PGNImportConfig, import_pgn


def test_import_pgn_writes_training_npz(tmp_path) -> None:
    pgn = tmp_path / "mini.pgn"
    pgn.write_text(
        textwrap.dedent(
            """
            [Event "Mini"]
            [Site "?"]
            [Date "2026.05.18"]
            [Round "1"]
            [White "A"]
            [Black "B"]
            [Result "1-0"]

            1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
            """
        ).strip()
        + "\n"
    )
    out = tmp_path / "out"
    written = import_pgn(PGNImportConfig(pgn=str(pgn), out=str(out), max_games=1, chunk_size=2))

    assert len(written) == 3
    raw = np.load(written[0])
    assert "actions" in raw
    assert "policies" not in raw
    dataset = SelfPlayDataset(out)
    sample = dataset[0]
    assert len(dataset) == 6
    assert sample["board"].shape == (NUM_INPUT_PLANES, 8, 8)
    assert sample["action"].ndim == 0
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["action"].shape == (2,)
    assert "policy" not in batch
    assert float(sample["value"]) == 1.0


def test_import_pgn_reads_zstd_archive(tmp_path) -> None:
    pgn_text = textwrap.dedent(
        """
        [Event "Compressed"]
        [Site "?"]
        [Date "2026.05.18"]
        [Round "1"]
        [White "A"]
        [Black "B"]
        [Result "1/2-1/2"]

        1. d4 d5 1/2-1/2
        """
    ).strip() + "\n"
    pgn = tmp_path / "mini.pgn.zst"
    pgn.write_bytes(zstandard.ZstdCompressor().compress(pgn_text.encode("utf-8")))

    out = tmp_path / "out-zst"
    written = import_pgn(PGNImportConfig(pgn=str(pgn), out=str(out), max_games=1))

    assert len(written) == 1
    dataset = SelfPlayDataset(out)
    assert len(dataset) == 2
    assert float(dataset[0]["value"]) == 0.0


def test_import_pgn_filters_by_min_elo(tmp_path) -> None:
    pgn = tmp_path / "rated.pgn"
    pgn.write_text(
        textwrap.dedent(
            """
            [Event "Low"]
            [Site "?"]
            [Date "2026.05.18"]
            [Round "1"]
            [White "A"]
            [Black "B"]
            [WhiteElo "1200"]
            [BlackElo "2400"]
            [Rated "True"]
            [Result "1-0"]

            1. e4 e5 1-0

            [Event "High"]
            [Site "?"]
            [Date "2026.05.18"]
            [Round "2"]
            [White "C"]
            [Black "D"]
            [WhiteElo "2300"]
            [BlackElo "2400"]
            [Rated "True"]
            [Result "0-1"]

            1. d4 Nf6 0-1
            """
        ).strip()
        + "\n"
    )
    out = tmp_path / "filtered"
    import_pgn(
        PGNImportConfig(
            pgn=str(pgn),
            out=str(out),
            min_elo=2200,
            rated_only=True,
            max_imported_games=1,
        )
    )

    summary = (out / "import_summary.txt").read_text()
    assert "games_seen=2" in summary
    assert "games_imported=1" in summary
    dataset = SelfPlayDataset(out)
    assert len(dataset) == 2
    assert float(dataset[0]["value"]) == -1.0


def test_rated_only_allows_archives_without_rated_header(tmp_path) -> None:
    pgn = tmp_path / "missing-rated.pgn"
    pgn.write_text(
        textwrap.dedent(
            """
            [Event "Archive"]
            [Site "?"]
            [Date "2026.05.18"]
            [Round "1"]
            [White "A"]
            [Black "B"]
            [WhiteElo "2100"]
            [BlackElo "2200"]
            [Result "1-0"]

            1. e4 e5 1-0
            """
        ).strip()
        + "\n"
    )
    out = tmp_path / "missing-rated-out"
    import_pgn(PGNImportConfig(pgn=str(pgn), out=str(out), min_elo=2000, rated_only=True))
    summary = (out / "import_summary.txt").read_text()
    assert "games_imported=1" in summary
