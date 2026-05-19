# Gated Self-Play 16 PV-Recent Probe

Date: 2026-05-19

## Run

`experiments/gated-selfplay16-pvrecent-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

This probe started from the PV-recent checkpoint because it is the only branch
that has shown occasional, unstable direct Stockfish draws at higher search. The
run used a small self-play batch and low self-play replay weight, with the direct
Stockfish promotion gate enabled.

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
  - `data/teacher/alpha_loss_gamelines_recent_v1`
- weights:
  - self-play: `0.10`
  - replay: `0.50 0.18 0.08 0.07 0.07`
- Stockfish gate:
  - `stockfish_gate_games=2`
  - `stockfish_gate_simulations=16`
  - `stockfish_gate_min_score=0.50`
  - `stockfish_gate_engine_time=0.05`

## Promotion

Candidate:

`experiments/gated-selfplay16-pvrecent-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `8.0/8`
- wins/draws/losses: `8/0/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/gated_selfplay16_pvrecent_stockfish_gate_16sims.pgn`
- promoted: `false`

The parent match looked strong, but the Stockfish gate rejected the checkpoint
immediately. The saved PGN shows the same direct-play tactical failures: one
loss allows a passed b-pawn and king attack, while the other loses material in
the opening after `...d4` and is quickly mated.

## Fixed Validation

Legal-policy validation on the rejected candidate:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5466` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3361` |
| `alpha_loss_reports_v2` | `0.6393` | `0.6393` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4561` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.7857` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2513` |
| `alpha_loss_gamelines_recent_v1` | n/a | `0.2721` |

The candidate improved the tiny poisoned-capture set and barely improved puzzle
lines, but it regressed broad Stockfish policy accuracy and recent PV-line
accuracy. That makes the 8/8 parent result look like opponent-specific drift
rather than a useful strength gain.

## Conclusion

Rejected. Low-weight self-play from the PV-recent branch can overtake its parent
in mirror evaluation, but the direct Stockfish gate and fixed validators show no
reliable progress toward direct-play strength.
