# Policy-Head-Only Broad-Low Probe

Date: 2026-05-19

## Run

`experiments/policyhead-broadlow-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This refined the first policy-head-only probe by lowering the learning rate and
using only broad policy replay:

- `policy_head_only=true`
- `value_weight=0.0`
- `lr=2e-6`
- `games=0`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
- replay weights: `0.70 0.30`

## Promotion

Candidate:

`experiments/policyhead-broadlow-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/16`
- wins/draws/losses: `0/0/16`
- promoted: `false`

The direct Stockfish gate did not run because the candidate failed the parent
gate.

## Fixed validation

Candidate-only validation used CPU, batch size 512, legal policy loss, and
`value_weight=0.25`. Parent qvalue metrics are included for comparison from the
same fixed validator.

| Dataset | Parent Policy Acc | Candidate Policy Acc | Delta |
| --- | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.6050` | `0.6042` | `-0.0007` |
| `puzzles/all_1200_2400_50k` | `0.3365` | `0.3389` | `+0.0023` |
| `alpha_loss_reports_v2` | `0.5738` | `0.5902` | `+0.0164` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4543` | `+0.0004` |
| `alpha_poisoned_captures_v2` | `0.0357` | `0.0357` | `+0.0000` |

Overall validation:

| Checkpoint | Policy Acc | Loss |
| --- | ---: | ---: |
| parent qvalue | `0.4198` | not rerun |
| candidate broad-low | `0.4208` | `2.3618` |

## Conclusion

Rejected. The lower-LR broad-only policy-head update preserved fixed broad
policy diagnostics, but search play collapsed completely against the parent.
Compared with the first policy-head-only probe, the small alpha-loss/poisoned
tail seems important for internal search stability, even though that mix
regressed broad fixed validation.
