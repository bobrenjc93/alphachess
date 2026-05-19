# Gated Self-Play 32 Qvalue Run

Date: 2026-05-19

## Run

`experiments/gated-selfplay32-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=32`
- `self_play_workers=4`
- `simulations=48`
- `epochs=1`
- `lr=1e-5`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- weights:
  - self-play: `0.35`
  - replay: `0.40 0.15 0.07 0.03`
- parent promotion gate: 8 games at 48 sims
- Stockfish gate enabled:
  - `stockfish_gate_games=2`
  - `stockfish_gate_simulations=16`
  - `stockfish_gate_min_score=0.50`
  - `stockfish_gate_engine_path=tools/stockfish/bin/stockfish`

## Self-play data

Generated 32 games in:

`experiments/gated-selfplay32-qvalue-vw025-material015/selfplay/iter_0001`

Self-play result summary:

- `1-0`: 16
- `0-1`: 5
- `1/2-1/2`: 9
- `*`: 2
- min plies: 37
- max plies: 180
- average plies: 105.84

## Promotion

Candidate:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/8`
- wins/draws/losses: `0/0/8`
- promoted: `false`

The direct Stockfish gate did not run because the candidate failed the parent gate first.

## Conclusion

Rejected. Adding 32 checkpoint self-play games with the current qvalue policy and replay mix degraded the candidate badly against its parent. Future self-play runs should either use a stronger incumbent, a stricter data-quality filter, or a lower self-play weight until the self-play policy is less noisy.
