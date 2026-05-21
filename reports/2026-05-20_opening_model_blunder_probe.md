# Opening Model-Blunder Probe

Timestamp: `2026-05-20T17:57:39-07:00`

## Summary

The FEN-branch source showed that tiny targeted repairs still overfit or fail
to move the branch targets. I broadened the same idea by mining
Stockfish-confirmed model-preferred bad moves from the current opening16k
teacher source.

This also exposed a CLI footgun: `HardNegativeConfig` and `ModelBlunderConfig`
default to `prefer_action_labels=True`, but the CLI's `store_true` flag made the
default `False`. I changed those mining CLIs to default to action labels and to
accept `--no-prefer-action-labels` when policy-target mining is intentional.

## Branch Model-Blunder Mine

Corrected action-target mine on the 42-position FEN branch source:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess model-blunders \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --data data/teacher/policyhead192_latest_failure_fens_branch_w2_p2_legalvalue_t10_v1 \
  --out data/teacher/policyhead192_latest_failure_fenbranch_modelblunders_actiontargets_top4_t10_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.1 \
  --max-positions 64 \
  --min-value-delta 0.08 \
  --bad-actions-per-position 4 \
  --batch-size 64 \
  --chunk-size 64 \
  --prefer-action-labels \
  --device cuda
```

Result:

- `42` positions seen
- `27` model-wrong positions
- `27` Stockfish-confirmed blunder positions
- `84` bad-action labels
- bad-action value drop: min `0.0806`, mean `0.3593`, max `0.7736`

## Opening16k Model-Blunder Mine

Broader action-target mine on the current 16k opening teacher source:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess model-blunders \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --data data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 \
  --out data/teacher/policyhead192_opening16k_modelblunders_actiontargets_top4_t05_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --max-positions 4096 \
  --min-value-delta 0.08 \
  --bad-actions-per-position 4 \
  --batch-size 512 \
  --chunk-size 1024 \
  --prefer-action-labels \
  --device cuda
```

Result:

- `4,096` positions seen
- `2,037` model-wrong positions
- `1,545` Stockfish-confirmed blunder positions
- `3,548` bad-action labels
- bad-action value drop: min `0.0802`, mean `0.2769`, max `1.5089`

Baseline validation of the current opening16k/stability `75%` blend:

| Slice | Top-1 | Top-3 | Top-5 | Policy loss | Bad-action loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad holdout, soft labels | `0.3398` | `0.5383` | `0.6395` | `3.6053` | `0.0000` |
| opening16k model blunders | `0.0000` | `0.4686` | `0.6395` | `3.3003` | `2.0078` |
| latest FEN branch model blunders | `0.0000` | `0.5185` | `0.7407` | `2.9801` | `1.6690` |

## Repair Attempts

I tried two policy-head-only repairs from the same parent. Both used the
opening16k model-blunder source plus the small FEN-branch model-blunder source,
with the original checkpoint as a policy-distillation anchor. Both were guarded
by the standard soft-label broad holdout floors
`holdout_policy_acc>=0.3395` and `holdout_policy_top3_acc>=0.5380`.

| Run | Key settings | Best useful target movement | Holdout result | Decision |
| --- | --- | --- | --- | --- |
| `opening16k_modelblunder_top4_distill3_lr1e6` | LR `1e-6`, distill weight `3.0`, bad-action weight `0.75`, max source repeat `8` | validation split top-1 reached only `0.0256`; bad-action loss stayed around `1.92` | epoch 1 already fell to soft holdout top-1/top-3 `0.3380`/`0.5339` | rejected before direct gate |
| `opening16k_modelblunder_top4_distill10_lr1e7` | LR `1e-7`, distill weight `10.0`, bad-action weight `0.50`, max source repeat `4` | bad-action loss edged down to `1.9125` by epoch 8 | every epoch missed the holdout guard; epoch 8 was `0.3362`/`0.5300` | rejected before direct gate |

The broader confirmed-blunder source is a better diagnostic than the six-FEN
branch source, but naive policy-head repair still damages broad ranking before
the model-blunder target becomes plausible. The next repair needs either a
different objective/schedule or a much broader balanced source, not more weight
on this slice.

## Verification

- `python3 -m compileall -q src/alpha_chess`: passed
- `git diff --check`: passed
- `uv run pytest tests/test_cli.py tests/test_model_blunders.py tests/test_hard_negatives.py -q`:
  `8 passed in 1.63s`
- `uv run pytest -q`: `127 passed in 95.42s`
