# Rootmate5 And 16k Refresh Follow-Up

Date: 2026-05-19

## Root-Mate-Depth Inference Check

Generated at:

- `2026-05-19T12:53:53-07:00` for the hard-label policy-head checkpoint
- `2026-05-19T12:53:55-07:00` for the broad policy-head checkpoint

Settings:

- `games=2`
- `simulations=16`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `root_mate_search_plies=5`
- `max_plies=180`

Results:

| Checkpoint | Score | W/D/L | PGN |
| --- | ---: | --- | --- |
| hard-label policy-head | `0.0/2` | `0/0/2` | `reports/policyhead_hardlabels_qvalue_rootmate5_stockfish_16sims.pgn` |
| broad policy-head | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_rootmate5_stockfish_16sims.pgn` |

## 16k Stockfish Plus Latest-Loss Policy Refresh

Generated at:

- `2026-05-19T13:02:42-07:00` for the manual direct Stockfish PGN

Run:

`experiments/policyhead-16k-leafloss-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `policy_head_only=true`
- `prefer_action_labels=true`
- `value_weight=0.0`
- `bad_action_weight=0.05`
- `lr=1e-5`
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

- `gpu-dev` A100 reservation `6fce3809`
- completed and copied back normally

Candidate:

`experiments/policyhead-16k-leafloss-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Promotion:

- parent match against qvalue: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`
- Stockfish gate skipped by the iteration driver because the candidate failed
  the parent match

Manual direct Stockfish check:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/policyhead_16k_leafloss_qvalue_stockfish_16sims.pgn`

## Conclusion

Rejected. A deeper root mate guard did not rescue direct play, and the broader
16k Stockfish plus latest-loss policy-head refresh regressed against the qvalue
parent before also losing the direct Stockfish check. The model is still not
direct-Stockfish-competitive or superhuman.
