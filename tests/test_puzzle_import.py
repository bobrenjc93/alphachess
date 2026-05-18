from alpha_chess.dataset import SelfPlayDataset
from alpha_chess.puzzle_import import PuzzleImportConfig, import_puzzles


def test_import_puzzles_writes_sparse_actions(tmp_path) -> None:
    csv_path = tmp_path / "puzzles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags",
                "p1,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1,c7c5 g1f3,1500,80,90,10,opening middlegame,https://lichess.org/abc,",
                "p2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1,c7c5 g1f3,2500,80,90,10,mate,https://lichess.org/def,",
            ]
        )
        + "\n"
    )
    out = tmp_path / "out"
    import_puzzles(
        PuzzleImportConfig(
            puzzles=str(csv_path),
            out=str(out),
            min_rating=1200,
            max_rating=2200,
            theme="opening",
        )
    )

    summary = (out / "puzzle_summary.txt").read_text()
    assert "rows_seen=2" in summary
    assert "rows_imported=1" in summary
    dataset = SelfPlayDataset(out)
    assert len(dataset) == 1
    assert dataset[0]["action"].ndim == 0
