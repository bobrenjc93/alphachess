# Policy Prior Temperature Probe

Date: 2026-05-19

## Change

Added `policy_prior_temperature` to MCTS prior normalization.

- Default `1.0` preserves existing behavior.
- Values above `1.0` flatten legal policy priors before PUCT selection.
- Invalid non-positive or non-finite values are rejected by `MCTSConfig`.
- Exposed through:
  - `alpha-chess self-play --policy-prior-temperature`
  - `alpha-chess eval --policy-prior-temperature`
  - `alpha-chess iterate --policy-prior-temperature`
  - `alpha-chess uci` / UCI `PolicyPriorTemperature`
  - `scripts/submit_gpu_iteration.sh`
  - `scripts/submit_gpu_training.sh`

The immediate motivation was the direct-loss diagnostic where a Stockfish-best move had a small but nonzero model prior and received too little root search. This knob makes that kind of exploration pressure testable without changing the checkpoint.

## Verification

- `python -m compileall -q src tests`
- `bash -n scripts/submit_gpu_iteration.sh scripts/submit_gpu_training.sh`
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_mcts.py tests/test_uci.py tests/test_iteration.py`
  - `22 passed`
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest`
  - `53 passed`

## Direct Stockfish smoke

Checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Common eval settings:

- opponent: `tools/stockfish/bin/stockfish`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `max_plies=180`
- CPU evaluation

Results:

| Prior temp | Sims | Games | Score | W/D/L | PGN |
| ---: | ---: | ---: | ---: | --- | --- |
| `2.0` | 16 | 2 | `0.0/2` | `0/0/2` | `reports/focus_pvlinesrecent_qvalue_priorT2_vs_stockfish_16sims.pgn` |
| `4.0` | 16 | 2 | `0.0/2` | `0/0/2` | `reports/focus_pvlinesrecent_qvalue_priorT4_vs_stockfish_16sims.pgn` |
| `4.0` | 64 | 1 | `0.0/1` | `0/0/1` | `reports/focus_pvlinesrecent_qvalue_priorT4_vs_stockfish_64sims.pgn` |

## Conclusion

The search knob works mechanically and is now available for future runs, but prior flattening alone did not improve direct Stockfish results on the recent PV-line checkpoint. This supports the earlier diagnosis that the issue is not only root prior starvation; the value estimates still mis-rank important continuations after exploration reaches them.
