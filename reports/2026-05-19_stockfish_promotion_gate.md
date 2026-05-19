# Stockfish Promotion Gate

Date: 2026-05-19

## Change

Added an optional direct-Stockfish gate to `alpha-chess iterate`.

The normal candidate-versus-parent promotion match still runs first. If the candidate passes that gate and `stockfish_gate_games > 0`, the iteration driver now runs a second evaluation against Stockfish and only promotes if the Stockfish score rate is at least `stockfish_gate_min_score`.

New iteration config fields and CLI flags:

- `--stockfish-gate-games`
- `--stockfish-gate-simulations`
- `--stockfish-gate-min-score`
- `--stockfish-gate-engine-path`
- `--stockfish-gate-engine-time`
- `--stockfish-gate-engine-depth`

Defaults:

- `stockfish_gate_games=0`, so existing runs are unchanged.
- if enabled without an explicit minimum, `stockfish_gate_min_score=0.50`.
- if gate simulations are omitted, the gate uses `eval_simulations`.

The GPU iteration helper now exposes matching environment variables:

- `STOCKFISH_GATE_GAMES`
- `STOCKFISH_GATE_SIMULATIONS`
- `STOCKFISH_GATE_MIN_SCORE`
- `STOCKFISH_GATE_ENGINE_PATH`
- `STOCKFISH_GATE_ENGINE_TIME`
- `STOCKFISH_GATE_ENGINE_DEPTH`

## Motivation

Recent targeted replay branches repeatedly promoted internally while losing direct Stockfish smokes. This gate prevents those false positives from becoming the league best checkpoint in future automated runs.

## Verification

- `bash -n scripts/submit_gpu_iteration.sh`
- `uv run alpha-chess iterate --help` exposes the new flags.
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_iteration.py`
  - `7 passed`
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest`
  - `55 passed`

## Conclusion

This is a process fix, not a strength breakthrough. It should make subsequent training runs more honest by requiring direct Stockfish evidence before promotion when the gate is enabled.
