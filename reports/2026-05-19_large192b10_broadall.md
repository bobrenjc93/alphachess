# Large 192x10 Broad-All Bootstrap Probe

Date: 2026-05-19

## Run

`experiments/bootstrap-large192b10-broadall-vw035-material015`

This followed the first 192x10 replay bootstrap by adding the broad puzzle set,
raising value loss weight, and training longer.

Config:

- `checkpoint=null`
- `channels=192`
- `blocks=10`
- `games=0`
- `epochs=4`
- `lr=2e-5`
- `value_weight=0.35`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/all_1200_2400_50k`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.35 0.25 0.20 0.08 0.08 0.04`

## Promotion

Candidate:

`experiments/bootstrap-large192b10-broadall-vw035-material015/checkpoints/iter_0001/latest.pt`

Null-baseline match:

- score: `2.0/4`
- wins/draws/losses: `0/4/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- simulations: `16`
- engine: `tools/stockfish/bin/stockfish`
- engine time: `0.05`
- PGN: `reports/bootstrap_large192b10_broadall_stockfish_gate_16sims.pgn`

Follow-up direct probe:

- score: `0.0/1`
- simulations: `64`
- PGN: `reports/bootstrap_large192b10_broadall_stockfish_64sims.pgn`

## Fixed Validation

Legal-policy validation, compared with the current PV-recent checkpoint and the
first 192x10 replay bootstrap:

| Dataset | PV-Recent Acc | First 192x10 Acc | Broad-All 192x10 Acc |
| --- | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5281` | `0.5051` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.1543` | `0.5385` |
| `alpha_loss_reports_v2` | `0.6393` | `0.3115` | `0.3115` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.2999` | `0.4396` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.5714` | `0.5714` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.5104` | `0.4935` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.3916` | `0.4648` |

The broader mix repaired the puzzle diagnostics and improved the game-line
teacher, but it reduced broad Stockfish-teacher accuracy and still failed every
direct Stockfish game.

## Conclusion

Rejected. The 192x10 from-scratch route can fit whichever replay source is
weighted heavily, but under this budget it does not produce a stronger player.
The direct PGNs still collapse tactically, so further large bootstrap runs
should not be prioritized without a better value/adversarial signal.
