# Gated Iter2 Broad16k Low-LR Probe

Date: 2026-05-19

## Run

`experiments/gated-iter2-broad16k-low`

Base checkpoint:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0002/latest.pt`

This tested whether the old 16k broad Stockfish replay set could help when applied conservatively from the stronger gated-selfplay iter2 checkpoint instead of from an earlier weaker checkpoint.

Config:

- `games=0`
- `epochs=1`
- `lr=3e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `experiments/gated-selfplay32-qvalue-vw025-material015/selfplay/iter_0001`
  - `data/teacher/stockfish_multipv_elo1800_16384`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_gamelines_recent_v1`
- replay weights: `0.05 0.55 0.15 0.10 0.10 0.05`
- Stockfish gate configured for 2 games at 16 sims, minimum score `0.50`

## Promotion

Candidate:

`experiments/gated-iter2-broad16k-low/checkpoints/iter_0001/latest.pt`

Parent:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0002/latest.pt`

Parent match:

- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

## Conclusion

Rejected. Applying the 16k broad Stockfish replay set conservatively from the gated iter2 checkpoint still weakened the parent matchup. More broad low-depth Stockfish replay remains a poor lever by itself.
