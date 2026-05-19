# Root King-Safety Filter Probe

Date: 2026-05-19

## Change Under Test

Added an optional root static-safety filter to MCTS. The filter evaluates root
candidate moves with shallow full-width search over a static score:

```text
material score + opponent king danger - own king danger
```

The king danger term counts attacks on the king ring with larger weights for
sliding pieces and queens, plus an explicit in-check penalty. This is intended
to catch slow king-safety collapses that are not immediate mate or material
losses.

New controls:

- `MCTSConfig.root_king_safety_search_plies`
- `MCTSConfig.root_king_safety_max_loss_cp`
- `alpha-chess self-play|eval|iterate|uci`
  `--root-king-safety-search-plies` and
  `--root-king-safety-max-loss-cp`
- `scripts/submit_gpu_iteration.sh`
  `ROOT_KING_SAFETY_SEARCH_PLIES` and
  `ROOT_KING_SAFETY_MAX_LOSS_CP`

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_mcts.py tests/test_evaluate.py tests/test_self_play.py tests/test_iteration.py tests/test_uci.py
38 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
79 passed
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
- `root_king_safety_search_plies=1`
- `max_plies=180`

| Checkpoint / Setting | Generated At | Score | W/D/L | PGN |
| --- | --- | ---: | --- | --- |
| qvalue, `root_king_safety_max_loss_cp=250` | `2026-05-19T13:24:07-07:00` | `0.0/2` | `0/0/2` | `reports/qvalue_rootkingsafety250_stockfish_16sims.pgn` |
| policyhead broad, `root_king_safety_max_loss_cp=250` | `2026-05-19T13:24:00-07:00` | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_rootkingsafety250_stockfish_16sims.pgn` |
| hard-label policyhead, `root_king_safety_max_loss_cp=250` | `2026-05-19T13:24:00-07:00` | `0.0/2` | `0/0/2` | `reports/policyhead_hardlabels_qvalue_rootkingsafety250_stockfish_16sims.pgn` |
| policyhead broad, `root_king_safety_max_loss_cp=100` | `2026-05-19T13:24:33-07:00` | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_rootkingsafety100_stockfish_16sims.pgn` |

## Conclusion

Rejected as a direct-play improvement. The root king-safety filter is available
as a search diagnostic and may be useful for future self-play filtering, but it
did not recover a Stockfish draw in these small gates. The failure remains too
deep or too broad for this shallow static safety term to fix.
