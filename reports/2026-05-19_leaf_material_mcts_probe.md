# Leaf-Material MCTS Probe

Date: 2026-05-19

## Change Under Test

Added an AutoGo-inspired leaf-value blend inside MCTS. When enabled, each
non-terminal MCTS leaf keeps the neural policy but blends the neural value with
the existing shallow material search from the leaf side-to-move perspective.

New controls:

- `MCTSConfig.leaf_material_value_weight`
- `MCTSConfig.leaf_material_search_plies`
- `alpha-chess self-play|eval|iterate|uci`
  `--leaf-material-value-weight` and `--leaf-material-search-plies`
- `scripts/submit_gpu_iteration.sh`
  `LEAF_MATERIAL_VALUE_WEIGHT` and `LEAF_MATERIAL_SEARCH_PLIES`

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_mcts.py tests/test_evaluate.py tests/test_self_play.py tests/test_iteration.py tests/test_uci.py
37 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
78 passed
```

## Direct Stockfish Probes

Settings unless noted:

- `games=2`
- `simulations=16`
- `engine_time=0.05`
- `workers=2`
- `device=cpu`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `max_plies=180`

| Checkpoint / Setting | Score | W/D/L | PGN |
| --- | ---: | --- | --- |
| hard-label policy head, `leaf_material_value_weight=0.5`, `leaf_material_search_plies=1` | `0.0/2` | `0/0/2` | `reports/policyhead_hardlabels_qvalue_leafmat050_p1_stockfish_16sims.pgn` |
| hard-label policy head, `leaf_material_value_weight=1.0`, `leaf_material_search_plies=1` | `0.0/2` | `0/0/2` | `reports/policyhead_hardlabels_qvalue_leafmat100_p1_stockfish_16sims.pgn` |
| hard-label policy head, `leaf_material_value_weight=1.0`, `leaf_material_search_plies=2` | `0.0/2` | `0/0/2` | `reports/policyhead_hardlabels_qvalue_leafmat100_p2_stockfish_16sims.pgn` |
| qvalue parent, `leaf_material_value_weight=0.5`, `leaf_material_search_plies=1` | `0.0/2` | `0/0/2` | `reports/qvalue_leafmat050_p1_stockfish_16sims.pgn` |
| qvalue parent, `leaf_material_value_weight=1.0`, `leaf_material_search_plies=1` | `0.0/2` | `0/0/2` | `reports/qvalue_leafmat100_p1_stockfish_16sims.pgn` |
| policyhead broad, `simulations=64`, no leaf blend | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_stockfish_64sims.pgn` |

## Conclusion

Rejected as a direct-play improvement. Leaf material mixing gives MCTS a
search-time tactical value fallback, but these direct games still collapse to
king attacks and forced mates. Increasing the previously best policyhead-broad
checkpoint from 16 to 64 simulations also did not recover a draw in this small
gate. The model is still not direct-Stockfish-competitive or superhuman.
