# Policy-Head-Only Bad-Action Probe

Date: 2026-05-19

## Run

`experiments/policyhead-badaction-qvalue-bw005-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This revisited bad-action supervision with the safer policy-head-only training
path:

- `policy_head_only=true`
- `value_weight=0.0`
- `bad_action_weight=0.05`
- `bad_action_margin=1.0`
- `lr=1e-5`
- `games=0`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_badactions_all_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.60 0.25 0.12 0.03`

## Promotion

Candidate:

`experiments/policyhead-badaction-qvalue-bw005-vw000-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `12.0/16`
- wins/draws/losses: `8/8/0`

Stockfish gate:

- score: `0.0/4`
- wins/draws/losses: `0/0/4`
- PGN: `reports/policyhead_badaction_qvalue_stockfish_gate_16sims.pgn`
- promoted: `false`

## Conclusion

Rejected. Policy-head-only bad-action supervision preserved enough search
behavior to beat the qvalue parent, but it did not improve the direct Stockfish
gate. The losses still show king attacks and material drops.

The bad-action signal is not the missing ingredient at this weight; the best
direct result remains the first policy-head-only broad run with `0.5/4` against
Stockfish.
