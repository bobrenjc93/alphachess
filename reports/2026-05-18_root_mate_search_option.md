# Root Mate Search Option

Date: 2026-05-18

## Change

Exposed the existing MCTS root mate-search depth as a configurable option:

```text
--root-mate-search-plies
```

Wired through:

```text
alpha-chess self-play
alpha-chess eval
alpha-chess iterate
alpha-chess uci
scripts/submit_gpu_iteration.sh via ROOT_MATE_SEARCH_PLIES
```

Default remains unchanged:

```text
root_mate_search_plies=3
```

## Verification

```text
uv run pytest
44 passed
```

CLI help exposes the option for `self-play`, `eval`, `iterate`, and `uci`.
`bash -n scripts/submit_gpu_iteration.sh` passes.

## Evaluation

Checkpoint:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Settings:

```text
material_value_weight=0.15
material_value_search_plies=2
root_mate_search_plies=5
root_material_search_plies=2
root_material_max_loss_cp=150
simulations=16
engine_time=0.05
```

Direct Stockfish result:

```text
score=0.0/2
wins=0
draws=0
losses=2
PGN=reports/focus_qvalue_rootmate5_material150_vs_stockfish_16sims.pgn
```

The option is useful for tactical search experiments and future training runs,
but deeper root mate search did not produce a direct Stockfish breakthrough in
this smoke.
