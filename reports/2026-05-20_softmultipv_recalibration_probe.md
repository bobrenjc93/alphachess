# Soft-MultiPV Recalibration Probe

Timestamp: `2026-05-20T00:01:38-07:00`

## Summary

Most recent fullnet192 policy-head runs used hard action labels from the
Stockfish teacher data. This probe tested whether a low-LR policy-head-only
recalibration on dense MultiPV policy targets could improve calibration without
more hard-negative margin pressure.

It did not. The branch regressed disjoint holdout top-1 from `0.3442` to
`0.3374`, so it was rejected before direct Stockfish play.

## Run

- Experiment:
  `experiments/policyhead192-softmultipv-recalibration-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Data:
  `data/teacher/stockfish_multipv_elo1800_65536_t005`,
  `data/teacher/stockfish_multipv_elo1800_8192_t05`,
  `data/teacher/stockfish_opening_elo1800_8192_t03`,
  `data/teacher/stockfish_multipv_elo2000_8192_t08`
- Data weights: `0.45 0.25 0.15 0.15`
- Epochs: `2`
- LR: `2e-6`
- `--prefer-action-labels`: omitted, so dense MultiPV policies were used.
- Selection metric: `holdout_policy_acc`
- Selected epoch: `1`

## Metrics

| Checkpoint | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss |
| --- | ---: | ---: | ---: | ---: |
| starting hard-negative broadrepair | `0.3442` | `0.5459` | `0.6464` | `2.7775` |
| soft-MultiPV epoch 1 | `0.3374` | `0.5377` | `0.6404` | `3.6342` |
| soft-MultiPV epoch 2 | `0.3372` | `0.5386` | `0.6403` | `3.5736` |

Selected training validation source metrics:

| Source | Policy acc | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| broad65k split | `0.7939` | N/A | N/A |
| t05 split | `0.5293` | N/A | N/A |
| opening split | `0.4922` | N/A | N/A |
| ELO2000 split | `0.4329` | N/A | N/A |

## Read

Soft targets at this stage pull the policy away from the hard-label move ranking
that the direct gate depends on. Dense MultiPV policy calibration may still be
useful earlier in training or with a different selection metric, but it is not a
good repair recipe for the current checkpoint.
