# Hard-Negative Top-k Repair

Timestamp: `2026-05-19T23:33:20-07:00`

## Summary

The last direct-loss diagnostic showed high-confidence raw policy blunders, so I
tested broader model-mined hard negatives on the 192-channel policy-head branch.
The first pass mined one bad action per top-1 teacher disagreement; the second
pass added support for up to three bad actions per mistaken position.

Neither candidate passed direct Stockfish. The best supervised result nudged the
disjoint broad holdout top-1 from `0.3439` to `0.3442`, but still scored `0.0/2`
against local Stockfish.

## Code change

- `alpha-chess hard-negatives --bad-actions-per-position N` now stores up to `N`
  wrong legal model moves for each position where the model top-1 differs from
  the teacher action.
- Training and validation now accept both old scalar `bad_actions` arrays and
  new padded vector `bad_actions` arrays.
- Focused tests passed:

```bash
uv run pytest tests/test_hard_negatives.py tests/test_model_and_data.py
```

Result: `26 passed`.

## Mined data

### Top-1 mined broad/t05 negatives

- Checkpoint:
  `experiments/policyhead192-broad65k-holdoutselect-v1/checkpoints/iter_0001/latest.pt`
- Data:
  `data/teacher/stockfish_multipv_elo1800_8192_t05`,
  `data/teacher/stockfish_multipv_elo1800_65536_t005`
- Output: `data/teacher/policyhead192_t05_broad65k_hardneg_v1`
- Positions: `73728`
- Hard-negative positions: `11290`
- Top-1 error rate: `0.153130`

### Top-3 mined broad/t05 negatives

- Checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Data:
  `data/teacher/stockfish_multipv_elo1800_8192_t05`,
  `data/teacher/stockfish_multipv_elo1800_65536_t005`
- Output: `data/teacher/policyhead192_t05_broad65k_hardneg_top3_v1`
- Positions: `73728`
- Hard-negative positions: `11162`
- Hard-negative actions: `33486`
- Top-1 error rate: `0.151394`

## Top-1 broad hard-negative repair

- Experiment:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-broad65k-holdoutselect-v1/checkpoints/iter_0001/latest.pt`
- Training data weights:
  broad65k `0.45`, t05 `0.25`, hard negatives `0.20`,
  fullnet loss slice `0.05`, latest loss slice `0.05`
- Epochs: `3`
- LR: `5e-6`
- Bad-action weight: `0.15`
- Selection metric: `holdout_policy_acc`
- Selected epoch: `2`

Selected validation:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| disjoint holdout | `0.3442` | `0.5459` | `0.6464` | N/A |
| mined broad/t05 hard negatives | `0.8486` | `0.9361` | `0.9568` | `3.2150` |
| t05 teacher | `0.5638` | `0.7446` | `0.8135` | N/A |
| fullnet loss slice | `0.2773` | `0.5042` | `0.6008` | `2.7146` |
| latest loss slice | `0.2653` | `0.4898` | `0.6531` | `2.8627` |

Direct gate:

| Check | Result | Artifact |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_hardneg_broadrepair_stockfish_gate.pgn` |

## Top-3 hard-negative repair

- Experiment:
  `experiments/policyhead192-hardneg-top3-repair-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Training data weights:
  broad65k `0.40`, t05 `0.25`, top-3 hard negatives `0.25`,
  fullnet loss slice `0.05`, latest loss slice `0.05`
- Epochs: `2`
- LR: `4e-6`
- Bad-action weight: `0.15`
- Selection metric: `holdout_policy_acc`
- Selected epoch: `1`

Selected epoch metrics:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| disjoint holdout | `0.3436` | `0.5470` | `0.6458` | N/A |
| epoch validation overall | `0.8457` | N/A | N/A | `2.4220` |
| top-3 hard-negative source | `0.8460` | N/A | N/A | `2.4212` |
| fullnet loss slice | `0.3077` | N/A | N/A | `0.7421` |
| latest loss slice | `0.4286` | N/A | N/A | `0.3491` |

Direct gate:

| Check | Result | Artifact |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_hardneg_top3_stockfish_gate.pgn` |

## Read

The top-1 hard-negative pass gives the best disjoint broad holdout top-1 seen so
far (`0.3442`), but the direct gate did not move. The top-3 pass proves the
new vector bad-action path works and substantially reduces bad-action loss, but
it trades away holdout top-1 and also fails direct play.

Conclusion: broader hard-negative pressure alone is still insufficient. The
next useful lever is likely stronger position selection or a better target than
single-move supervised labels, not simply more margin pressure on the same
broad teacher slice.
