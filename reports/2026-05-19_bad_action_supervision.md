# Bad-Action Supervision Probe

Date: 2026-05-19

## Change

Added optional bad-action supervision for blunder replay:

- `stockfish-teacher` now stores `bad_actions` and `played_moves` for sampled
  PGN positions where `--min-value-delta` identifies the played move as a value
  drop.
- `SelfPlayDataset` loads and collates `bad_actions` when present.
- `train`, `validate`, and `iterate` accept `--bad-action-weight` and
  `--bad-action-margin`.
- training adds a soft margin term that pushes the played bad action below the
  teacher target action.
- `scripts/submit_gpu_iteration.sh` forwards `BAD_ACTION_WEIGHT` and
  `BAD_ACTION_MARGIN`.

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_stockfish_teacher.py tests/test_model_and_data.py tests/test_iteration.py
25 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
60 passed
```

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_badactions_all_v1`

Summary:

- sources: all report PGNs
- games_seen: `165`
- games_used: `165`
- positions: `706`
- files: `99`
- valid bad actions: `695`
- `engine_time=0.04`
- `min_value_delta=0.08`
- `player_name=AlphaChess`
- `multipv=4`
- `policy_temperature_cp=200`
- `position_stride=1`
- `pv_plies=0`
- `game_line_plies=0`

Parent diagnostic on the new bad-action set:

- policy_acc: `0.2167`
- bad_action_loss with weight `0.5`: `2.2330`

## Training Probe

Run:

`experiments/focus-badactions-pvrecent-bw050-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=4e-6`
- `value_weight=0.25`
- `bad_action_weight=0.50`
- `bad_action_margin=1.0`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_badactions_all_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.52 0.16 0.14 0.04 0.08 0.06`

## Result

Candidate:

`experiments/focus-badactions-pvrecent-bw050-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/8`
- wins/draws/losses: `0/0/8`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

Fixed validation:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.4641` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3229` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5410` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4466` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6786` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2760` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2798` |
| `alpha_loss_badactions_all_v1` | `0.2167` | `0.2564` |

Bad-action diagnostic loss improved from `2.2330` to `1.5768`, but the weight
was far too disruptive: broad Stockfish policy accuracy collapsed and the
candidate lost every parent game.

## Lower-Weight Follow-Up

Run:

`experiments/focus-badactionslow-pvrecent-bw005-vw025-material015`

Changed from the first probe:

- `lr=2e-6`
- `bad_action_weight=0.05`
- bad-action replay share lowered from `0.14` to `0.04`
- broad Stockfish replay raised from `0.52` to `0.58`

Parent match:

- score: `4.0/8`
- wins/draws/losses: `4/0/4`

Stockfish gate:

- 16 simulations, 2 games
- score: `0.0/2`
- PGN: `reports/focus_badactionslow_pvrecent_bw005_stockfish_gate_16sims.pgn`

Higher-search smoke:

- 64 simulations, 1 game
- score: `0.0/1`
- PGN: `reports/focus_badactionslow_pvrecent_bw005_s64_vs_stockfish.pgn`

Fixed validation:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5217` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3309` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5902` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4527` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6786` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2682` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2827` |
| `alpha_loss_badactions_all_v1` | `0.2167` | `0.2181` |

Bad-action diagnostic loss moved only slightly, from `2.2330` to `2.0660`, while
broad Stockfish policy accuracy still fell by more than six points.

## Conclusion

Keep the bad-action supervision machinery, but reject both initial weightings.
Even as a small regularizer, this signal trades away too much broad policy
strength and does not improve the direct Stockfish gate.
