# Opening MultiPV12 Probe

Date: 2026-05-19

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_opening_multipv12_v1`

Summary:

- sources: `reports/*.pgn`
- games_seen: `163`
- games_used: `163`
- positions: `2048`
- files: `98`
- average nonzero policy moves: approximately `11.46`
- `engine_time=0.04`
- `min_value_delta=None`
- `player_name=AlphaChess`
- `multipv=12`
- `policy_temperature_cp=200`
- `position_stride=1`
- `min_ply=0`
- `max_ply=24`

This was intended to be a richer opening repair signal than the sparse
opening-only bad-action set.

## Run

`experiments/focus-openingmultipv12-pvrecent-low-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `iterations=4`
- `games=0`
- `epochs=1`
- `lr=1e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `eval_games=8`
- `eval_simulations=48`
- `stockfish_gate_games=2`
- `stockfish_gate_simulations=16`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_opening_multipv12_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.54 0.18 0.06 0.04 0.10 0.08`

## Result

All candidates were rejected:

| Iteration | Parent Match | W/D/L | Stockfish Gate | PGN |
| --- | ---: | ---: | ---: | --- |
| `1` | `6.0/8` | `4/4/0` | `0.0/2` | `reports/focus_openingmultipv12_pvrecent_low_iter1_stockfish_gate_16sims.pgn` |
| `2` | `8.0/8` | `8/0/0` | `0.0/2` | `reports/focus_openingmultipv12_pvrecent_low_iter2_stockfish_gate_16sims.pgn` |
| `3` | `0.0/8` | `0/0/8` | not run | |
| `4` | `6.0/8` | `4/4/0` | `0.0/2` | `reports/focus_openingmultipv12_pvrecent_low_iter4_stockfish_gate_16sims.pgn` |

Higher-search smoke on the strongest parent-match candidate:

- candidate: `experiments/focus-openingmultipv12-pvrecent-low-vw025-material015/checkpoints/iter_0002/latest.pt`
- score: `0.0/1`
- simulations: `64`
- PGN: `reports/focus_openingmultipv12_pvrecent_low_iter2_s64_vs_stockfish.pgn`
- result: lost as White to `13...Qh3#`

Fixed legal-policy validation for `iter_0002`:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5747` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3309` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5082` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4541` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.7143` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2643` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2778` |
| `alpha_loss_opening_multipv12_v1` | `0.3721` | `0.3809` |

Overall validation:

- parent: `val_policy_acc=0.4133`, `val_loss=2.3608`
- candidate: `val_policy_acc=0.4127`, `val_loss=2.3505`

## Conclusion

Rejected. The richer opening MultiPV target gives a small targeted improvement
on the new opening set and can dominate the PV-recent parent in self-match, but
it still scores `0/2` in every direct Stockfish gate that runs and loses the
64-simulation smoke game. The validation profile also shows a large regression
on `alpha_loss_reports_v2`, so this looks like another supervised opening repair
that improves local imitation without fixing the direct tactical failure mode.
