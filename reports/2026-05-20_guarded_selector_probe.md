# Guarded Composite Selector Probe

Timestamp: `2026-05-20T12:53:06-07:00`

## Summary

I added guarded composite checkpoint selection and ran a short broad-only
policy-head continuation to test the failure mode that showed up in the engine
self-play probe: policy loss can improve while broad ranking metrics regress.
The selector behaved as intended. No epoch met the broad holdout top-3 floor, so
the run produced epoch checkpoints but no selected `latest.pt`, and I did not
spend a direct Stockfish gate on it.

## Parent Baseline

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess validate \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --batch-size 512 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --device cuda
```

Parent broad holdout:

| Metric | Value |
| --- | ---: |
| policy loss | `3.7873` |
| top-1 | `0.3395` |
| top-3 | `0.5421` |
| top-5 | `0.6425` |

## Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-broad65k-guarded-composite-v1/checkpoints/iter_0001 \
  --epochs 5 \
  --batch-size 512 \
  --lr 1e-7 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3394' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

The command ended with:

```text
ValueError: No epoch satisfied select_best_by 'holdout_policy_acc+holdout_policy_top3_acc' with requirements holdout_policy_acc>=0.3394 holdout_policy_top3_acc>=0.5420
```

## Epoch Metrics

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3392` | `0.5388` | `0.6398` | `3.5951` | no |
| 2 | `0.3387` | `0.5386` | `0.6401` | `3.6139` | no |
| 3 | `0.3395` | `0.5386` | `0.6404` | `3.6365` | no |
| 4 | `0.3392` | `0.5381` | `0.6395` | `3.5842` | no |
| 5 | `0.3392` | `0.5386` | `0.6396` | `3.5962` | no |

## Read

This is a useful negative result. The broad-only continuation lowered policy
loss by about `0.15-0.20`, but it damaged top-3/top-5 ranking on the disjoint
holdout. That is exactly the kind of checkpoint the earlier single-metric or
loss-led workflow could accidentally gate. The guarded selector should be used
for the next mixed-source GPU run so the direct Stockfish gate only sees
checkpoints that preserve broad ranking metrics.
