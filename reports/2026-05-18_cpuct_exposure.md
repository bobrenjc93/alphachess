# C-PUCT Exposure

Date: 2026-05-18

## Change

Exposed the MCTS exploration constant:

```text
--c-puct
```

Wired through:

```text
alpha-chess self-play
alpha-chess eval
alpha-chess iterate
alpha-chess uci
scripts/submit_gpu_iteration.sh via C_PUCT
```

Default remains unchanged:

```text
c_puct=1.5
```

## Verification

```text
uv run pytest
46 passed
```

`bash -n scripts/submit_gpu_iteration.sh` passes, and CLI help exposes
`--c-puct` for `eval` and `iterate`.

## Evaluation

Checkpoint:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Settings:

```text
material_value_weight=0.15
material_value_search_plies=2
simulations=16
engine_time=0.05
```

Direct Stockfish sweep:

```text
c_puct=0.5 score=0.0/2
PGN=reports/focus_qvalue_cpuct05_vs_stockfish_16sims.pgn

c_puct=2.5 score=0.0/2
PGN=reports/focus_qvalue_cpuct25_vs_stockfish_16sims.pgn
```

The knob is now available for training and UCI experiments, but this small sweep
did not produce a direct Stockfish breakthrough.
