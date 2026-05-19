# Gated Self-Play Iter2 Poison-Fix Probe

Date: 2026-05-19

## Run

`experiments/gated-selfplay32-iter2-poisonfix`

Base checkpoint:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0002/latest.pt`

The base candidate beat qvalue internally and improved broad Stockfish policy accuracy, but failed the direct Stockfish gate and regressed poisoned/game-line diagnostics. This follow-up raised poisoned and game-line replay pressure while preserving broad Stockfish replay.

Config:

- `games=0`
- `epochs=1`
- `lr=5e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `experiments/gated-selfplay32-qvalue-vw025-material015/selfplay/iter_0001`
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_gamelines_recent_v1`
- replay weights: `0.05 0.50 0.18 0.08 0.12 0.07`
- Stockfish gate configured for 2 games at 16 sims, minimum score `0.50`

## Promotion

Candidate:

`experiments/gated-selfplay32-iter2-poisonfix/checkpoints/iter_0001/latest.pt`

Parent:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0002/latest.pt`

Parent match:

- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

## Conclusion

Rejected. The poisoned/game-line correction from the iter2 checkpoint cost too much parent strength. The useful signal remains the gated-selfplay iter2 checkpoint's broad policy improvement, but it still fails direct play.
