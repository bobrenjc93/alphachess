# Opening/ELO2000/Tactical Mix Probe

Timestamp: `2026-05-19T23:49:30-07:00`

## Summary

The hard-negative probes suggested that margin pressure alone is not enough, so
I tested a small policy-head-only mix aimed at sources that match the recent
direct failures: opening positions, stronger ELO2000 Stockfish labels, and
tactical continuation/loss-line replay.

The branch improved the targeted source metrics but regressed the disjoint broad
holdout and still scored `0.0/2` against local Stockfish. It is rejected.

## Baseline validation

Checkpoint:
`experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| opening 8192 t03 | `0.5157` | `0.7787` | `0.8656` | N/A |
| ELO2000 6719 t08 | `0.4309` | `0.6229` | `0.7135` | N/A |
| tactical recent tiebreak | `0.2686` | `0.4711` | `0.5908` | `1.7720` |
| disjoint holdout | `0.3442` | `0.5459` | `0.6464` | N/A |

## Repair run

- Experiment:
  `experiments/policyhead192-opening-elo2000-tactical-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Training data weights:
  broad65k `0.30`, t05 `0.18`, opening `0.20`, ELO2000 `0.17`,
  tactical recent `0.05`, fullnet loss slice `0.05`, latest loss slice `0.05`
- Epochs: `2`
- LR: `3e-6`
- Bad-action weight: `0.10`
- Selection metric: `holdout_policy_acc`
- Selected epoch: `1`

Selected epoch metrics:

| Source | Policy acc | Notes |
| --- | ---: | --- |
| disjoint holdout | `0.3425` | Regressed from `0.3442`. |
| opening source split | `0.5278` | Improved from baseline `0.5157`. |
| ELO2000 source split | `0.3972` | Training split metric; not directly comparable to full baseline. |
| tactical recent source split | `0.2917` | Improved from baseline `0.2686`. |

Direct gate:

| Check | Result | Artifact |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_opening_elo2000_tactical_stockfish_gate.pgn` |

## Read

The opening and tactical slices are learnable, but the mix pulls the policy away
from the disjoint broad holdout and still does not stop direct tactical losses.
This points toward needing better tactical state coverage or a different
training target, not just adding small specialist slices to the same
policy-head-only recipe.
