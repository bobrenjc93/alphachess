# Value-Only Self-Play 64 Qvalue Run

Date: 2026-05-19

## Change Under Test

This run used source policy loss weights from `1cc3c74` so generated self-play
contributed value loss but not policy loss:

- `self_play_weight=0.10`
- `self_play_policy_weight=0.0`
- replay policy weights defaulted to `1.0`

The intent was to keep the outcome/value signal from self-play without training
on the sparse 64-simulation visit-count policies.

## Run

`experiments/gated-selfplay64-valueonlypolicy-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=64`
- `self_play_workers=8`
- `simulations=64`
- `max_plies=180`
- `epochs=1`
- `batch_size=128`
- `lr=4e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.55 0.20 0.10 0.05`
- parent promotion gate: 16 games at 64 sims, 4 eval workers
- Stockfish gate: 4 games at 16 sims

## Self-play data

Generated 64 games in:

`experiments/gated-selfplay64-valueonlypolicy-qvalue-vw025-material015/selfplay/iter_0001`

Self-play result summary:

- `1-0`: 26
- `0-1`: 21
- `1/2-1/2`: 15
- `*`: 2
- positions: 6,884
- min plies: 43
- max plies: 180
- average plies: 107.56
- average nonzero policy entries: 3.93

This is the same deterministic self-play batch shape as the prior policy-bearing
64-game run.

## Promotion

Candidate:

`experiments/gated-selfplay64-valueonlypolicy-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `12.0/16`
- wins/draws/losses: `8/8/0`

Stockfish gate:

- score: `0.0/4`
- wins/draws/losses: `0/0/4`
- PGN: `reports/gated_selfplay64_valueonlypolicy_qvalue_stockfish_gate_16sims.pgn`
- promoted: `false`

Higher-search direct smoke:

- 64 simulations, 1 Stockfish game
- score: `0.0/1`
- PGN: `reports/gated_selfplay64_valueonlypolicy_qvalue_stockfish_64sims.pgn`

## Fixed validation

Validation used CPU, batch size 512, legal policy loss, and `value_weight=0.25`.

| Dataset | Examples | Parent Policy Acc | Candidate Policy Acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | 4,096 | `0.6050` | `0.5469` | `-0.0581` |
| `puzzles/all_1200_2400_50k` | 50,000 | `0.3365` | `0.3344` | `-0.0022` |
| `alpha_loss_reports_v2` | 61 | `0.5738` | `0.6393` | `+0.0656` |
| `puzzles/lines_1200_2400_100k` | 100,000 | `0.4539` | `0.4533` | `-0.0006` |
| `alpha_poisoned_captures_v2` | 28 | `0.0357` | `0.2857` | `+0.2500` |
| self-play `iter_0001` | 6,884 | `0.6034` | `0.6031` | `-0.0003` |

Overall validation:

| Checkpoint | Policy Acc | Loss |
| --- | ---: | ---: |
| parent qvalue | `0.4276` | `2.3287` |
| candidate value-only | `0.4252` | `2.3220` |

## Conclusion

Rejected. Removing self-play policy loss fixed the parent-match collapse: this
candidate beat the qvalue parent `12/16`, unlike the policy-bearing 64-game run
that scored `4/16`. The direct Stockfish gate still failed `0/4`, and the
64-simulation smoke also lost.

The result supports keeping source policy weights as an experiment control, but
value-only self-play is not sufficient by itself. It improves internal search
matchups while direct play still blunders material and mate threats.
