# No-Tree-Reuse Probe

Date: 2026-05-19

## Change

Added an explicit tree-reuse toggle for self-play, evaluation, iteration, and the
GPU iteration wrapper.

- Default behavior remains tree reuse enabled.
- `alpha-chess self-play`, `alpha-chess eval`, and `alpha-chess iterate` now
  accept `--no-tree-reuse`.
- `scripts/submit_gpu_iteration.sh` accepts `TREE_REUSE=0` / `false` and passes
  `--no-tree-reuse`.

This isolates whether retained subtree visit/value statistics are contributing
to the direct Stockfish failures without changing existing experiment defaults.

## Verification

Focused tests:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_evaluate.py tests/test_self_play.py tests/test_iteration.py
15 passed
```

Full suite:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
73 passed
```

## Direct Stockfish Probe

Checkpoint:

`experiments/policyhead-broad-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Eval settings:

- `--no-tree-reuse`
- `simulations=16`
- `games=2`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- CPU evaluation

Result:

| Games | Score | W/D/L | PGN |
| ---: | ---: | --- | --- |
| 2 | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_notreereuse_stockfish_16sims.pgn` |

Conclusion: fresh-root search does not recover the latest policy-head broad
checkpoint. The failures remain tactical rather than a simple stale-tree issue.
