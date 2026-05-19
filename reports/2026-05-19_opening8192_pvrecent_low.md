# Conservative Opening-Window PV-Recent Probe

Date: 2026-05-19

## Run

`experiments/focus-opening8192-pvrecent-low-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

This is a conservative PV-recent follow-up to the earlier qvalue-start
opening-window probe. It keeps broad Stockfish replay dominant, lowers LR, and
uses the existing opening-window Stockfish teacher as a secondary signal.

Config:

- `games=0`
- `epochs=1`
- `lr=2e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/teacher/stockfish_opening_elo1800_8192_t03`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.52 0.12 0.14 0.04 0.10 0.08`

## Promotion

Candidate:

`experiments/focus-opening8192-pvrecent-low-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `4.0/8`
- wins/draws/losses: `0/8/0`

Stockfish gate:

- 16 simulations, 2 games
- score: `0.0/2`
- PGN: `reports/focus_opening8192_pvrecent_low_stockfish_gate_16sims.pgn`
- promoted: `false`

Higher-search smoke:

- 64 simulations, 1 game
- score: `0.0/1`
- PGN: `reports/focus_opening8192_pvrecent_low_s64_vs_stockfish.pgn`

## Fixed Validation

Legal-policy validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5654` |
| `stockfish_opening_elo1800_8192_t03` | `0.3807` | `0.3779` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3296` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5902` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4537` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.7143` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2630` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2825` |

The opening-window replay did not improve the opening-window diagnostic from
the PV-recent base and regressed broad Stockfish and old Alpha-loss accuracy.

## Conclusion

Rejected. Conservative opening-window replay from the PV-recent branch produces
a drawish parent match but does not improve direct Stockfish play or fixed
opening diagnostics.
