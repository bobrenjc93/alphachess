# Policy-Head-Only Strong Tactical-Tail Probe

Date: 2026-05-19

## Run

`experiments/policyhead-tailstrong-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This kept the successful policy-head-only learning rate from the first
policy-head run, but increased the tactical tail:

- `policy_head_only=true`
- `value_weight=0.0`
- `lr=1e-5`
- `games=0`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.55 0.25 0.15 0.05`

## Promotion

Candidate:

`experiments/policyhead-tailstrong-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `12.0/16`
- wins/draws/losses: `8/8/0`

Stockfish gate:

- score: `0.0/4`
- wins/draws/losses: `0/0/4`
- PGN: `reports/policyhead_tailstrong_qvalue_stockfish_gate_16sims.pgn`
- promoted: `false`

## Conclusion

Rejected. The stronger tactical-tail mix beat the qvalue parent more clearly
than the first policy-head-only run, but direct Stockfish play regressed from
`0.5/4` to `0/4`. The losses still show mating attacks and material drops.

The best policy-head-only result remains
`experiments/policyhead-broad-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`,
which at least drew one Stockfish gate game.
