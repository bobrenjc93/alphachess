# Full-Network Hard-Label 16k Leaf-Loss Probe

Date: 2026-05-19

## Run

`experiments/fullhard-16k-leafloss-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This tested whether a conservative full-network update could use the hard
Stockfish action labels and latest-loss data better than the recent
policy-head-only attempts.

Config:

- `policy_head_only=false`
- `prefer_action_labels=true`
- `value_weight=0.25`
- `bad_action_weight=0.02`
- `lr=2e-6`
- `epochs=2`
- `games=0`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_16384`
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v4`
  - `data/teacher/alpha_loss_pvlines_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_leafmcts_v1`
- replay weights: `0.45 0.15 0.15 0.08 0.05 0.02 0.10`

GPU runner:

- `gpu-dev` A100 reservation `16c97cbc`
- completed and copied back normally

Candidate:

`experiments/fullhard-16k-leafloss-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Training checkpoint metrics:

```text
epoch_loss=2.1858
val_loss=2.1583
val_policy_loss=1.9824
val_policy_acc=0.4472
val_source_6_policy_acc=0.2969
val_source_6_bad_action_loss=0.8947
```

## Promotion

Parent match against qvalue:

- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`

The iteration driver skipped the Stockfish gate because the candidate failed
the parent match.

Manual direct Stockfish check:

- generated at `2026-05-19T13:17:29-07:00`
- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/fullhard_16k_leafloss_qvalue_stockfish_16sims.pgn`

## Conclusion

Rejected. A full-network low-LR hard-label update did not recover the
regression seen in policy-head-only 16k/latest-loss tuning. It underperformed
the qvalue parent and still lost direct Stockfish games. The latest-loss slice
is not yet a reliable improvement signal for either policy-head-only or
full-network updates.
