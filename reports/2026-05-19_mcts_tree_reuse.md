# MCTS Tree Reuse Probe

Date: 2026-05-19

## Change

Added subtree reuse across moves in evaluation and self-play:

- `AlphaZeroMCTS.run()` can continue from an existing root node.
- `advance_root()` keeps the subtree reached by the played move when available.
- root mate/material filters are reapplied when a reused node becomes the new
  root.
- eval games advance model and opponent roots after every move.
- self-play advances the root after each sampled move.

This is closer to standard AlphaZero search behavior and gives later moves
access to analysis already spent under the prior root.

## Verification

Focused tests:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_mcts.py tests/test_evaluate.py tests/test_self_play.py
15 passed
```

Full suite:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
58 passed
```

## Direct Stockfish Smoke

Checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Eval settings:

- opponent: `tools/stockfish/bin/stockfish`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `max_plies=180`
- CPU evaluation

Results:

| Sims | Games | Score | W/D/L | PGN |
| ---: | ---: | ---: | --- | --- |
| 16 | 2 | `0.0/2` | `0/0/2` | `reports/tree_reuse_pvrecent_stockfish_16sims.pgn` |
| 64 | 1 | `0.0/1` | `0/0/1` | `reports/tree_reuse_pvrecent_stockfish_64sims.pgn` |

## Conclusion

Kept as a search implementation improvement, but rejected as a direct-play fix.
Tree reuse does not recover the PV-recent checkpoint's Stockfish gate; the games
still fail through tactical and positional collapse.
