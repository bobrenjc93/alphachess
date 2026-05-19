# Material Quiescence Value Prior

Date: 2026-05-18

## Change

Added an opt-in quiescent material value prior for neural evaluators:

```text
--material-value-search-plies N
```

When `--material-value-weight` is nonzero and search plies are enabled, the
material component evaluates short capture/promotion sequences instead of only
static material. Defaults remain unchanged:

```text
material_value_weight=0.0
material_value_search_plies=0
```

Wired through:

- `alpha-chess self-play`
- `alpha-chess eval`
- `alpha-chess iterate`
- UCI option `MaterialValueSearchPlies`
- `scripts/submit_gpu_iteration.sh` via `MATERIAL_VALUE_SEARCH_PLIES`

## Verification

```text
uv run pytest
42 passed
```

`bash -n scripts/submit_gpu_iteration.sh` passes, and `alpha-chess eval --help`
exposes `--material-value-search-plies`.

## Evaluation

Checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Material weight `0.15`, quiescence search plies `2`.

Stockfish smoke at 16 simulations:

```text
games=2
score=0.5
wins=0
draws=1
losses=1
pgn=reports/material_qsearch2_puzzlelines20_vs_stockfish_16sims.pgn
```

Uniform sanity at 32 simulations:

```text
games=4
score=3.0
wins=2
draws=2
losses=0
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/material_qsearch2_puzzlelines20_s64_vs_stockfish.pgn
```

Material weight `0.25`, quiescence search plies `2`, Stockfish at 16
simulations:

```text
games=2
score=0.0
wins=0
draws=0
losses=2
pgn=reports/material_qsearch2_w025_puzzlelines20_vs_stockfish_16sims.pgn
```

## Conclusion

The quiescent material prior produced a nonzero Stockfish result for the
current best checkpoint, but it also made uniform play more drawish and did not
survive the 64-simulation Stockfish sample. It is useful as an opt-in inference
and self-play experiment, not a completed playing-strength breakthrough.
