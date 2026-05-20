# Guarded Composite Selector Probe

Timestamp: `2026-05-20T13:09:24-07:00`

## Summary

I added guarded composite checkpoint selection and ran two short policy-head
continuations to test the failure mode that showed up in the engine self-play
probe: policy loss or top-1 can improve while broad top-k ranking regresses.
The selector behaved as intended. Neither run met the broad holdout top-3 floor,
so they produced epoch checkpoints but no selected `latest.pt`, and I did not
spend a direct Stockfish gate on either.

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

## Mixed-Source Follow-Up

I also tried a low-LR mixed-source policy-head pass with the same broad holdout
floors:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 data/teacher/alpha_recent80_firstblunder_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-guarded-mix-v1/checkpoints/iter_0001 \
  --epochs 5 \
  --batch-size 512 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.70 0.10 0.10 0.10 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3394' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

This also ended with no selected epoch.

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3398` | `0.5377` | `0.6390` | `3.6270` | no |
| 2 | `0.3396` | `0.5381` | `0.6390` | `3.6228` | no |
| 3 | `0.3397` | `0.5380` | `0.6385` | `3.6281` | no |
| 4 | `0.3395` | `0.5377` | `0.6393` | `3.6207` | no |
| 5 | `0.3395` | `0.5378` | `0.6388` | `3.6243` | no |

This is sharper than the broad-only run: top-1 was at or slightly above the
parent baseline in several epochs, but top-3 stayed well below the `0.5420`
floor. That confirms top-1 alone is not a sufficient selector for the current
teacher mix.

## Blend Follow-Up

I then blended the parent with the best-looking rejected epochs. This is cheap
and directly tests whether a small step toward the candidate can keep the
parent's top-k ranking while taking the lower-loss/top-1 signal.

| Source checkpoint | Weight on candidate | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mixed-source epoch 1 | `0.05` | `0.3398` | `0.5420` | `0.6422` | `3.7775` | near guard |
| mixed-source epoch 1 | `0.10` | `0.3397` | `0.5421` | `0.6417` | `3.7680` | preserves top-3 |
| mixed-source epoch 1 | `0.20` | `0.3398` | `0.5421` | `0.6412` | `3.7496` | preserves top-3 |
| mixed-source epoch 1 | `0.30` | `0.3401` | `0.5406` | `0.6403` | `3.7321` | top-3 regresses |
| broad-only epoch 3 | `0.05` | `0.3402` | `0.5422` | `0.6421` | `3.7779` | clears guard |
| broad-only epoch 3 | `0.10` | `0.3401` | `0.5426` | `0.6422` | `3.7688` | best guard-passing blend |
| broad-only epoch 3 | `0.20` | `0.3401` | `0.5414` | `0.6410` | `3.7512` | top-3 regresses |
| broad-only epoch 3 | `0.30` | `0.3398` | `0.5410` | `0.6407` | `3.7345` | top-3 regresses |

The `broad-only epoch 3` blend at `0.10` is the best checkpoint from this
sequence: it improves disjoint broad top-1 from `0.3395` to `0.3401`, top-3
from `0.5421` to `0.5426`, keeps top-5 roughly flat, and lowers policy loss.
I started a two-game direct Stockfish gate for it, but the persistent GPU
reservation was canceled mid-run, so no complete PGN/result was produced. This
blend should be the next direct-gate candidate when the persistent workspace is
available again.
