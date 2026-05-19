# Bounded Root Material Guard

Date: 2026-05-18

## Change

The optional root material guard now scores root moves with a bounded material
minimax instead of only capture/check quiescence. The minimax expands captures,
promotions, and checks; if no tactical move exists, it considers up to 8 quiet
moves ranked by the highest-value opponent piece they newly attack. Leaf nodes
still get a one-ply material quiescence extension.

Default search is unchanged because the guard only runs when
`root_material_search_plies > 0`.

## Verification

The first full-width version was too slow (`tests/test_mcts.py` took 57s), so it
was replaced with the bounded quiet-threat candidate set.

```text
uv run pytest tests/test_mcts.py  # 8 passed in 1.90s
uv run pytest                     # 49 passed in 4.77s
```

## Direct Play

Checkpoint:

```text
experiments/focus-pvlines-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

Settings:

```text
material_value_weight=0.15
material_value_search_plies=2
root_material_search_plies=2
root_material_max_loss_cp=150
simulations=16
```

Result:

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_pvlines_qvalue_boundedguard150_vs_stockfish_16sims.pgn
```

## Conclusion

The bounded guard is fast enough to keep as an opt-in tactical search tool, but
it is not a direct Stockfish breakthrough. The failure mode remains deeper
king-safety and tactical-sequence evaluation, not just immediate root material
loss.
