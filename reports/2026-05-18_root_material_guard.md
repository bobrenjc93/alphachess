# Root Material Guard

Date: 2026-05-18

## Change

Added an opt-in root material safety filter for MCTS. After the existing root
mate filter, the guard can run a short capture/promotion quiescence search from
each root move and prune moves whose resulting material score drops too far
below the current static material score.

Configuration:

```text
root_material_search_plies=0  # default off
root_material_max_loss_cp=250
```

Wired through:

- `alpha-chess self-play`
- `alpha-chess eval`
- `alpha-chess iterate`
- UCI options
- `scripts/submit_gpu_iteration.sh` via `ROOT_MATERIAL_SEARCH_PLIES` and
  `ROOT_MATERIAL_MAX_LOSS_CP`

The guard is off by default because the first smoke tests did not improve
playing strength and the extra root search slows low-simulation evaluation.

## Tests

```text
uv run pytest
40 passed
```

Additional coverage verifies:

- the guard prunes a move that exposes a queen to a rook
- the guard can be disabled
- the guard prunes the `...Qxa1` queen-for-rook trap seen in a Stockfish PGN
- UCI root material options parse correctly

`bash -n scripts/submit_gpu_iteration.sh` also passes.

## Evaluation

Checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`

Material value weight: `0.15`

Opt-in guard:

```text
--root-material-search-plies 2
--root-material-max-loss-cp 250
```

Uniform sanity at 16 simulations:

```text
games=2
score=1.5
wins=1
draws=1
losses=0
```

Stockfish smoke at 16 simulations:

```text
games=2
score=0.0
wins=0
draws=0
losses=2
pgn=reports/root_material_guard250_vs_stockfish_16sims.pgn
```

The earlier loose `350cp` calibration also scored `0/2` and is saved at
`reports/root_material_guard_vs_stockfish_16sims.pgn`.

## Conclusion

The root material guard catches concrete one-move material traps, but it is not
enough for the current Stockfish gate. The PGNs still show tactical collapse
over several forcing moves, so this should remain an opt-in experiment while
the next branch targets deeper tactical search or stronger tactical/value
supervision.
