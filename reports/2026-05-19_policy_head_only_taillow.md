# Policy-Head-Only Tactical-Tail Low-LR Probe

Date: 2026-05-19

## Run

`experiments/policyhead-taillow-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This refined the first policy-head-only broad probe by keeping the same replay
mix but lowering the learning rate:

- `policy_head_only=true`
- `value_weight=0.0`
- `lr=4e-6`
- `games=0`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.65 0.25 0.08 0.02`

## Promotion

Candidate:

`experiments/policyhead-taillow-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `4.0/16`
- wins/draws/losses: `0/8/8`
- promoted: `false`

The direct Stockfish gate did not run because the candidate failed the parent
gate.

## Conclusion

Rejected. Lowering the policy-head-only learning rate from `1e-5` to `4e-6`
with the same tactical-tail replay mix reduced internal strength: the earlier
run drew all 16 parent games and scored `0.5/4` against Stockfish, while this
run failed the parent gate at `4/16`.

The promising point remains the original `1e-5` policy-head-only branch, not
this lower-LR variant.
