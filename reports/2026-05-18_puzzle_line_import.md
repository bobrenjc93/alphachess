# Puzzle Line Import

Date: 2026-05-18

## Change

Added an opt-in Lichess puzzle importer mode:

```text
--include-solution-line
```

When enabled, the importer writes every legal move in the puzzle solution line
instead of only the first move. Values are signed from the original winning
side:

- winning side to move: `+value`
- defending side to move: `-value`

The default behavior remains first-move-only for compatibility.

## Generated Data

```text
out=data/puzzles/lines_1200_2400_100k
source=data/raw/lichess_db_puzzle.csv.zst
rows_seen=34324
rows_imported=19840
rows_skipped=14484
positions=100000
files=25
min_rating=1200
max_rating=2400
include_solution_line=true
```

Value distribution:

```text
positive=50000
negative=50000
zero=0
```

The data directory is an ignored local training artifact.

## Verification

```text
uv run pytest
41 passed
```

`alpha-chess import-puzzles --help` exposes `--include-solution-line`.

## Next Use

Use this dataset as a tactical sequence replay source instead of increasing
the weight of the older first-move-only puzzle set.
