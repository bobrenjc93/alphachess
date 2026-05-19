# Check-Aware Quiescence

Date: 2026-05-18

## Change

Extended the material quiescence move generator to include checking moves in
addition to captures and promotions. This affects:

```text
material_value(..., search_plies=N)
root_material_search_plies=N
```

Quiet checking moves are ordered ahead of ordinary material moves so mate checks
are considered quickly.

## Verification

```text
uv run pytest
45 passed
```

Added a regression check using the Fool's Mate position:

```text
Qh4# is a quiet checking move
material_value(board, search_plies=1) == 1.0
```

## Evaluation

Checkpoint:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Direct Stockfish with check-aware material value:

```text
material_value_weight=0.15
material_value_search_plies=2
simulations=16
score=0.0/2
PGN=reports/focus_qvalue_checkqsearch_vs_stockfish_16sims.pgn
```

Direct Stockfish with check-aware material value plus root material guard:

```text
material_value_weight=0.15
material_value_search_plies=2
root_material_search_plies=2
root_material_max_loss_cp=150
simulations=16
score=0.0/2
PGN=reports/focus_qvalue_checkqsearch_rootmaterial150_vs_stockfish_16sims.pgn
```

The change improves tactical coverage for quiet mate checks, but the current
checkpoint still fails direct Stockfish smokes.
