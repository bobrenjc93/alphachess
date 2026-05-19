# Elo2000 Stockfish Teacher PV-Recent Probe

Date: 2026-05-19

## Data

Generated ignored local replay data:

`data/teacher/stockfish_multipv_elo2000_8192_t08`

Summary:

- source: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- games_seen: `121332`
- games_used: `176`
- positions: `6719`
- files: `27`
- `min_elo=2000`
- `min_initial_seconds=180`
- `engine_time=0.08`
- `multipv=4`
- `policy_temperature_cp=180`
- `position_stride=2`

## Run

`experiments/focus-elo2000sf-pvrecent-low-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=2e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo2000_8192_t08`
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.55 0.15 0.12 0.04 0.08 0.06`

## Promotion

Candidate:

`experiments/focus-elo2000sf-pvrecent-low-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

## Fixed Validation

Legal-policy validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo2000_8192_t08` | `0.3367` | `0.3181` |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.4595` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3228` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5410` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4508` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6786` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2591` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2812` |

The candidate regressed the new high-Elo teacher diagnostic as well as the
standard broad Stockfish diagnostic.

## Conclusion

Rejected. A deeper high-Elo Stockfish teacher from the old Lichess archive does
not provide a usable fine-tune direction for the PV-recent checkpoint at this
scale.
