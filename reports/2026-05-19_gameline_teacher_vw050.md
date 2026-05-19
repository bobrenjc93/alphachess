# Game-Line Teacher Value Probe

Date: 2026-05-19

## Code change

Added `alpha-chess stockfish-teacher --game-line-plies`.

The existing `--pv-plies` option labels positions along Stockfish's preferred continuation. The new option labels positions reached by the actual PGN moves after each sampled root, which is intended to train the value head on the child states Alpha actually entered in its loss games.

Verification:

- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_stockfish_teacher.py`
  - `8 passed`
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest`
  - `54 passed`

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_gamelines_recent_v1`

Summary:

- sources: 29 Stockfish-smoke PGNs from qvalue, process-worker, PV-line, value-weight, opening, and prior-temperature branches
- games_seen: `52`
- games_used: `52`
- positions: `1470`
- files: `29`
- `engine_time=0.03`
- `min_value_delta=0.08`
- `player_name=AlphaChess`
- `multipv=4`
- `policy_temperature_cp=200`
- `position_stride=1`
- `pv_plies=4`
- `game_line_plies=2`

## Training run

Run:

`experiments/focus-gamelines-qvalue-vw050-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=1.2e-5`
- `value_weight=0.50`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_gamelines_recent_v1`
- replay weights: `0.45 0.15 0.08 0.04 0.28`

Promotion gate versus qvalue baseline:

- 48 simulations, 8 games
- score: `8.0/8`
- wins/draws/losses: `8/0/0`
- promoted internally

Candidate checkpoint:

`experiments/focus-gamelines-qvalue-vw050-material015/checkpoints/iter_0001/latest.pt`

## Fixed validation

Legal-policy validation:

| Dataset | Policy Acc |
| --- | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5647` |
| `puzzles/all_1200_2400_50k` | `0.3323` |
| `alpha_loss_reports_v2` | `0.5574` |
| `puzzles/lines_1200_2400_100k` | `0.4476` |
| `alpha_poisoned_captures_v2` | `0.5000` |
| `alpha_loss_pvlines_recent_v1` | `0.2305` |
| `alpha_loss_gamelines_recent_v1` | `0.3068` |

This branch regressed broad Stockfish policy accuracy and the poisoned-capture diagnostic relative to the current qvalue/PV-line candidates.

## Direct Stockfish smoke

Eval settings:

- opponent: `tools/stockfish/bin/stockfish`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `max_plies=180`
- CPU evaluation

Results:

| Sims | Games | Score | W/D/L | PGN |
| ---: | ---: | ---: | --- | --- |
| 16 | 2 | `0.0/2` | `0/0/2` | `reports/focus_gamelines_qvalue_vw050_vs_stockfish_16sims.pgn` |
| 64 | 1 | `0.0/1` | `0/0/1` | `reports/focus_gamelines_qvalue_vw050_vs_stockfish_64sims.pgn` |

## Conclusion

Rejected. Labeling Alpha's actual game-line child states produced a very strong internal promotion result, but did not improve direct Stockfish play and hurt several fixed diagnostics. The result reinforces that the current promotion gate is too weak for targeted replay branches.
