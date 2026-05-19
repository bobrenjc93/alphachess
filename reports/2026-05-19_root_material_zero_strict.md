# Root Material Zero-Loss Fix

Date: 2026-05-19

## Change

Fixed root-material filtering so `root_material_max_loss_cp=0` means "allow no
material loss" instead of disabling the filter. The filter is still disabled by
setting `root_material_search_plies=0`.

The previous strict root-material smoke was invalid because `_filter_root_material`
returned early for `max_loss_cp <= 0`.

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_mcts.py tests/test_evaluate.py tests/test_uci.py
20 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
64 passed
```

## Direct Stockfish Smoke

Checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Eval settings:

- opponent: `tools/stockfish/bin/stockfish`
- `engine_time=0.05`
- `material_value_weight=0.30`
- `material_value_search_plies=2`
- `root_material_search_plies=3`
- `root_material_max_loss_cp=0`
- `max_plies=180`
- CPU evaluation

Results:

| Sims | Games | Score | W/D/L | PGN |
| ---: | ---: | ---: | --- | --- |
| 16 | 2 | `0.0/2` | `0/0/2` | `reports/pvrecent_fixed_strict_rootmaterial_stockfish_16sims.pgn` |
| 64 | 1 | `0.0/1` | `0/0/1` | `reports/pvrecent_fixed_strict_rootmaterial_stockfish_64sims.pgn` |

## Conclusion

Keep the fix. It makes strict root-material experiments meaningful, but strict
material filtering still does not recover the PV-recent direct Stockfish gate.
The losses remain tactical and include mating attacks after apparently quiet
opening play.
