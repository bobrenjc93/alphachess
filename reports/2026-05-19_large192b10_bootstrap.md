# Large 192x10 Replay Bootstrap Probe

Date: 2026-05-19

## Run

`experiments/bootstrap-large192b10-replay-vw025-material015`

This probe trained a larger network from replay only, rather than fine-tuning
the current 128-channel checkpoints.

Config:

- `checkpoint=null`
- `channels=192`
- `blocks=10`
- `games=0`
- `epochs=2`
- `lr=2e-5`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.50 0.25 0.10 0.10 0.05`

## Promotion

Candidate:

`experiments/bootstrap-large192b10-replay-vw025-material015/checkpoints/iter_0001/latest.pt`

Null-baseline match:

- score: `3.0/4`
- wins/draws/losses: `2/2/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- simulations: `16`
- engine: `tools/stockfish/bin/stockfish`
- engine time: `0.05`
- PGN: `reports/bootstrap_large192b10_stockfish_gate_16sims.pgn`

Follow-up direct probe:

- score: `0.0/1`
- simulations: `64`
- PGN: `reports/bootstrap_large192b10_stockfish_64sims.pgn`

## Fixed Validation

Legal-policy validation, compared with the current PV-recent checkpoint for
orientation:

| Dataset | PV-Recent Acc | Large Bootstrap Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5281` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.1543` |
| `alpha_loss_reports_v2` | `0.6393` | `0.3115` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.2999` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.5714` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.5104` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.3916` |

The larger replay-only model learned the recent PV/game-line targets better
than the PV-recent checkpoint, but it gave up too much broad puzzle and
Stockfish-teacher accuracy.

## Conclusion

Rejected. A 192-channel, 10-block replay-only bootstrap is not a useful direct
strength improvement under this short training budget. The run is useful mainly
as evidence that the recent failure-line corpora can be fit by a larger model,
but direct Stockfish play still collapses tactically.
