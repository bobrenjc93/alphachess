# Opening-Window Teacher Blend Probe

Date: 2026-05-19

## Change under test

- Added Stockfish teacher ply windows with `--min-ply` / `--max-ply`.
- Generated an opening-window replay set from early Lichess positions only:
  - `data/teacher/stockfish_opening_elo1800_8192_t03`
  - 8192 positions, 8 shards
  - source `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
  - `min_elo=1800`, `min_initial_seconds=180`
  - `engine_time=0.03`, `multipv=4`, `policy_temperature_cp=200`
  - `min_ply=0`, `max_ply=20`, `position_stride=1`
  - 400 games used from 8983 games seen

## Training run

Run: `experiments/focus-opening8192-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=1.5e-5`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/teacher/stockfish_opening_elo1800_8192_t03`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
- replay weights: `0.40 0.25 0.15 0.08 0.04 0.08`

Promotion gate versus base qvalue checkpoint:

- 48 simulations, 8 games
- score: `4.0/8`
- wins/draws/losses: `4/0/4`
- promoted at the configured `0.50` threshold

Candidate checkpoint:

`experiments/focus-opening8192-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

## Fixed validation

Legal-policy validation:

| Dataset | Loss |
| --- | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5701` |
| `stockfish_opening_elo1800_8192_t03` | `0.3997` |
| `puzzles/all_1200_2400_50k` | `0.3313` |
| `alpha_loss_reports_v2` | `0.5410` |
| `puzzles/lines_1200_2400_100k` | `0.4532` |
| `alpha_poisoned_captures_v2` | `0.6429` |
| `alpha_loss_pvlines_recent_v1` | `0.2396` |

Relative to the broad qvalue baseline, the branch improved the recent PV replay loss, but regressed the broad Stockfish replay and did not materially improve old puzzle-line policy loss.

## Direct Stockfish smoke

Eval config:

- opponent: `tools/stockfish/bin/stockfish`
- `engine_time=0.05`
- `max_plies=180`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- CPU evaluation

Results:

| Sims | Games | Score | W/D/L | PGN |
| ---: | ---: | ---: | --- | --- |
| 16 | 2 | `0.0/2` | `0/0/2` | `reports/focus_opening8192_qvalue_vs_stockfish_16sims.pgn` |
| 64 | 1 | `0.0/1` | `0/0/1` | `reports/focus_opening8192_qvalue_vs_stockfish_64sims.pgn` |

## Conclusion

Rejected. The opening-window replay blend can pass the internal qvalue promotion gate, but it loses direct Stockfish games at both 16 and 64 simulations. The fixed validation profile also shows a broad Stockfish replay regression versus the current qvalue baseline, so the branch is not a better candidate for the main line.
