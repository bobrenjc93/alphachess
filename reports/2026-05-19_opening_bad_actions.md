# Opening Bad-Action Probe

Date: 2026-05-19

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_opening_badactions_v2`

Summary:

- sources: `reports/*.pgn`
- games_seen: `180`
- games_used: `159`
- positions: `399`
- files: `104`
- valid bad actions: `398`
- `engine_time=0.04`
- `min_value_delta=0.08`
- `player_name=AlphaChess`
- `multipv=4`
- `policy_temperature_cp=200`
- `position_stride=1`
- `min_ply=0`
- `max_ply=24`

Parent diagnostic on the new opening bad-action set:

- policy_acc: `0.2231`
- bad_action_loss with weight `0.02`: `2.0184`

## Run

`experiments/focus-openingbadactions-pvrecent-bw002-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=1e-6`
- `value_weight=0.25`
- `bad_action_weight=0.02`
- `bad_action_margin=1.0`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_opening_badactions_v2`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.58 0.18 0.02 0.04 0.10 0.08`

## Result

Candidate:

`experiments/focus-openingbadactions-pvrecent-bw002-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/2`
- simulations: `16`
- PGN: `reports/focus_openingbadactions_pvrecent_bw002_stockfish_gate_16sims.pgn`

Higher-search smoke:

- score: `0.0/1`
- simulations: `64`
- PGN: `reports/focus_openingbadactions_pvrecent_bw002_s64_vs_stockfish.pgn`

Fixed validation:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5310` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3309` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5574` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4529` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.7500` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2630` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2825` |
| `alpha_loss_opening_badactions_v2` | `0.2231` | `0.2281` |

The opening bad-action loss improved only slightly, from `2.0184` to `1.9281`.

## Conclusion

Rejected. A tiny opening bad-action signal can beat the PV-recent parent in the
internal match and preserve most targeted replay diagnostics, but it still
regresses broad Stockfish policy accuracy and fails direct Stockfish at both 16
and 64 simulations.
