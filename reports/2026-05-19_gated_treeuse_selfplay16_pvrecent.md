# Gated Tree-Reuse Self-Play 16 PV-Recent Probe

Date: 2026-05-19

## Run

`experiments/gated-treeuse-selfplay16-pvrecent-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

This repeats the earlier low-weight PV-recent self-play probe after adding MCTS
tree reuse, so both self-play generation and promotion evaluation can carry
subtrees forward across moves.

Config:

- `games=16`
- `self_play_workers=4`
- `simulations=48`
- `eval_games=8`
- `eval_simulations=48`
- `epochs=1`
- `lr=6e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- weights:
  - self-play: `0.10`
  - replay: `0.50 0.18 0.08 0.07 0.07`
- Stockfish gate: 2 games at 16 sims, minimum score `0.50`

## Promotion

Candidate:

`experiments/gated-treeuse-selfplay16-pvrecent-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/gated_treeuse_selfplay16_pvrecent_stockfish_gate_16sims.pgn`
- promoted: `false`

Higher-search smoke:

- 64 simulations, 1 Stockfish game
- score: `0.0/1`
- PGN: `reports/gated_treeuse_selfplay16_pvrecent_s64_vs_stockfish.pgn`

## Fixed Validation

Legal-policy validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5469` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3341` |
| `alpha_loss_reports_v2` | `0.6393` | `0.6557` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4554` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.7857` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2487` |
| `alpha_loss_gamelines_recent_v1` | `0.2612` | `0.2612` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2854` |

This is materially similar to the pre-tree-reuse self-play16 probe: targeted
diagnostics improve, broad Stockfish policy accuracy regresses by about four
points, and direct Stockfish play remains at `0.0`.

## Conclusion

Rejected. Tree reuse is useful infrastructure, but re-running the small
PV-recent self-play recipe under tree reuse does not produce a stronger direct
player.
