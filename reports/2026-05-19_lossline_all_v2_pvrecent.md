# Refreshed Loss-Line Teacher PV-Recent Probe

Date: 2026-05-19

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_gamelines_all_v2`

Summary:

- sources: all `90` report PGNs
- games_seen: `122`
- games_used: `122`
- positions: `4096`
- files: `73`
- `engine_time=0.04`
- `min_value_delta=0.08`
- `player_name=AlphaChess`
- `multipv=4`
- `policy_temperature_cp=200`
- `position_stride=1`
- `pv_plies=4`
- `game_line_plies=3`

This refresh expands the prior recent game-line teacher from `1470` positions
over `52` loss games to `4096` positions over `122` AlphaChess-vs-Stockfish
games, including the latest gated, blend, material-sweep, and deeper-search
failures.

## Run

`experiments/focus-lossallv2-pvrecent-vw035-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=4e-6`
- `value_weight=0.35`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.52 0.16 0.06 0.04 0.10 0.12`
- Stockfish gate: 2 games at 16 sims, minimum score `0.50`

Candidate:

`experiments/focus-lossallv2-pvrecent-vw035-material015/checkpoints/iter_0001/latest.pt`

## Promotion

Parent match:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/focus_lossallv2_pvrecent_vw035_stockfish_gate_16sims.pgn`
- promoted: `false`

Higher-search smoke:

- 64 simulations, 1 Stockfish game
- score: `0.0/1`
- PGN: `reports/focus_lossallv2_pvrecent_vw035_s64_vs_stockfish.pgn`

## Fixed Validation

Legal-policy validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5881` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3313` |
| `alpha_loss_reports_v2` | `0.6393` | `0.6230` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4534` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6786` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2721` |
| `alpha_loss_gamelines_recent_v1` | `0.2612` | `0.2667` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2883` |

The refreshed teacher slightly improves broad Stockfish, recent PV-line, and
game-line policy accuracy, but it regresses poisoned-capture and older
Alpha-loss diagnostics. The direct Stockfish games still collapse tactically,
including queen/king attacks in both gate games and a forced mate in the 64-sim
smoke.

## Conclusion

Rejected. The larger, fresher loss-line dataset gives a stronger parent match
and modest targeted-diagnostic gains, but it does not produce direct-play
strength. The Stockfish promotion gate remains the right blocker for these
targeted replay branches.
