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

## Continuation: Lower Self-Play Weight

Reused the same 32 generated self-play games with a lower self-play weight:

- `games=0`
- `self_play_weight=0.10`
- replay weights: `0.55 0.20 0.10 0.05`
- `lr=6e-6`
- same qvalue base checkpoint
- same Stockfish gate: 2 games, 16 sims, minimum score `0.50`

Candidate:

`experiments/gated-selfplay32-qvalue-vw025-material015/checkpoints/iter_0002/latest.pt`

Parent match:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- local reproduction PGN: `reports/gated_selfplay32_qvalue_iter2_stockfish_gate_16sims.pgn`
- promoted: `false`

Fixed validation:

| Dataset | Policy Acc |
| --- | ---: |
| `stockfish_multipv_elo1800_4096` | `0.6143` |
| `puzzles/all_1200_2400_50k` | `0.3363` |
| `alpha_loss_reports_v2` | `0.6066` |
| `puzzles/lines_1200_2400_100k` | `0.4556` |
| `alpha_poisoned_captures_v2` | `0.3929` |
| `alpha_loss_gamelines_recent_v1` | `0.2442` |

The lower self-play mix produced a candidate that beat the qvalue parent and improved broad Stockfish policy accuracy, but the Stockfish promotion gate correctly rejected it. Direct play remains the limiting signal.

Additional inference-only check:

- `root_mate_search_plies=5`
- 16 simulations, 2 Stockfish games
- score: `0.0/2`
- PGN: `reports/gated_selfplay32_qvalue_iter2_rootmate5_stockfish_16sims.pgn`

Deepening the existing root mate filter did not recover the direct-play gate.
