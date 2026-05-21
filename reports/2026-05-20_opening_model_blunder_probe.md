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

## Interpolation Follow-Up

Timestamp: `2026-05-20T18:07:43-07:00`

I checked whether the rejected repairs contain a useful low-weight direction by
interpolating them back into the parent checkpoint and validating against the
standard soft-label broad holdout plus both model-blunder slices.

| Blend | Broad top-1 | Broad top-3 | Opening-blunder top-1 | Opening bad-action loss | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| distill10 epoch 8, `5%` | `0.3392` | `0.5385` | `0.0013` | `2.0064` | missed broad top-1 floor |
| distill10 epoch 8, `10%` | `0.3389` | `0.5380` | `0.0045` | `2.0052` | missed broad top-1 floor |
| distill10 epoch 8, `20%` | `0.3387` | `0.5360` | `0.0065` | `2.0028` | missed both broad floors |
| distill3 epoch 1, `2.5%` | `0.3396` | `0.5385` | `0.0006` | `2.0075` | broad-safe but target movement is negligible |
| distill3 epoch 1, `5%` | `0.3395` | `0.5387` | `0.0013` | `2.0072` | just missed broad top-1 floor |
| distill3 epoch 1, `10%` | `0.3394` | `0.5385` | `0.0013` | `2.0067` | missed broad top-1 floor |

The only broad-safe blend moves the opening-blunder slice by less than one tenth
of a percentage point and barely changes bad-action loss, so I did not spend a
direct Stockfish gate on it. Interpolation confirms that the current
model-blunder repair direction is too weak per unit of broad-holdout damage.

## Value-Delta Weighted Objective

Timestamp: `2026-05-20T18:16:52-07:00`

I added optional value-delta weighting for bad-action margin loss:

- datasets now load aligned `bad_action_deltas` from Stockfish teacher files
  and aligned `value_deltas` from model-blunder files;
- `train` and `validate` now accept `--bad-action-delta-weight`;
- the default is unchanged at `0.0`.

The first guarded policy-head probe used the same opening16k and FEN-branch
model-blunder sources as above, with `--bad-action-delta-weight 3.0`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-data data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 \
  --policy-distill-weight 10.0 \
  --data data/teacher/policyhead192_opening16k_modelblunders_actiontargets_top4_t05_v1 data/teacher/policyhead192_latest_failure_fenbranch_modelblunders_actiontargets_top4_t10_v1 \
  --bad-action-weight 0.5 \
  --bad-action-delta-weight 3.0 \
  --select-best-require 'holdout_policy_acc>=0.3395' 'holdout_policy_top3_acc>=0.5380'
```

Result: no epoch satisfied the broad holdout guard. Epoch 1 already fell to
soft-label holdout top-1/top-3 `0.3380`/`0.5339`; by epoch 8 the training
validation split reached only top-1/top-3 `0.0321`/`0.5256`. No direct gate was
spent.

## Bad-Action Probability Objective

Timestamp: `2026-05-20T18:24:51-07:00`

The margin objective still forces the teacher target above each bad move. I
added `--bad-action-loss-type probability` as a gentler alternative that
directly suppresses the probability of listed bad moves and lets policy
distillation decide where the probability mass should go. The default remains
`margin`.

First probe:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-data data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 \
  --policy-distill-weight 10.0 \
  --data data/teacher/policyhead192_opening16k_modelblunders_actiontargets_top4_t05_v1 data/teacher/policyhead192_latest_failure_fenbranch_modelblunders_actiontargets_top4_t10_v1 \
  --source-policy-weights 0.0 0.0 \
  --bad-action-weight 10.0 \
  --bad-action-loss-type probability \
  --bad-action-delta-weight 3.0 \
  --select-best-require 'holdout_policy_acc>=0.3395' 'holdout_policy_top3_acc>=0.5380'
```

Result: no epoch satisfied the broad holdout guard. The run followed the same
ranking pattern as the margin probes: epoch 1 fell to holdout top-1/top-3
`0.3380`/`0.5338`, and epoch 8 reached only training validation top-1/top-3
`0.0321`/`0.5321`. The probability objective changes the bad-action loss scale
but does not solve the broad-transfer problem on this source.

## Lower-Pressure Probability Follow-Up

Timestamp: `2026-05-20T18:38:19-07:00`

I repeated the probability objective with `--bad-action-weight 1.0` instead of
`10.0`, keeping the same model-blunder sources, `--bad-action-delta-weight 3.0`,
and the same soft broad-holdout guard:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
  --distill-data data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 \
  --policy-distill-weight 10.0 \
  --data data/teacher/policyhead192_opening16k_modelblunders_actiontargets_top4_t05_v1 data/teacher/policyhead192_latest_failure_fenbranch_modelblunders_actiontargets_top4_t10_v1 \
  --bad-action-weight 1.0 \
  --bad-action-loss-type probability \
  --bad-action-delta-weight 3.0 \
  --select-best-require 'holdout_policy_acc>=0.3395' 'holdout_policy_top3_acc>=0.5380'
```

No epoch satisfied the soft broad-holdout guard. Epoch 1 reproduced the same
soft holdout top-1/top-3 as the higher-pressure run, `0.3380`/`0.5338`.
Validation with hard action labels gave a less pessimistic broad read,
`0.3433`/`0.5388`, and opening-blunder top-1 moved from `0.0000` to `0.0181`,
so I spent one comparable two-game direct check on epoch 1:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/opening16k_modelblunder_top4_prob_delta3_bw1_distill10_lr1e6/epoch_0001.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 4 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 data/teacher/policyhead192_mat4_latest_fullgame_context_t05_v1 \
  --good-action-book-top-k 2 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_modelblunder_prob_bw1_epoch1_top2_mat4_stockfish_gate.pgn
```

Result: `0.0/2`. The White game repeated the `Bc4` tactical failure family,
and the Black game repeated an adjacent `...Kh8` kingside-collapse family. The
lower-pressure probability objective is therefore rejected as a direct-strength
repair, not just as a soft-holdout miss.

## Evaluation PGN Timestamps

Timestamp: `2026-05-20T18:38:19-07:00`

Evaluation PGNs now write real provenance headers: `Date`, `Time`, and
`AlphaChessTimestamp` with an ISO timestamp. This keeps future direct-check
artifacts aligned with the README's real-timestamp progress tracker instead of
relying only on filesystem mtimes.

## Fresh Context Blend

Timestamp: `2026-05-20T18:52:50-07:00`

I mined first-blunder context from the lower-pressure probability direct losses:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/policyhead192_modelblunder_prob_bw1_epoch1_top2_mat4_stockfish_gate.pgn \
  --out data/teacher/policyhead192_modelblunder_prob_bw1_firstblunders_context_t05_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.08 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --first-blunder-only \
  --blunder-context-plies 2 \
  --pv-plies 4 \
  --game-line-plies 2 \
  --chunk-size 256
```

Result: `18` positions from `2` failed games. The probability epoch was slightly
worse than the parent on this fresh slice (`2.3485` vs `2.3304` policy loss;
`0.0177` vs `0.0172` probability bad-action loss), confirming that the
probability repair did not learn these adjacent failures.

I then folded the fresh source into the broader opening/stability repair mix.
The raw repair again missed the hard-label broad guard, but a `25%` blend back
into the parent was broad-safe:

| Checkpoint | Holdout top-1 | Holdout top-3 | Fresh top-1/top-3 | Fresh bad-action margin |
| --- | ---: | ---: | ---: | ---: |
| parent | `0.3456` | `0.5432` | `0.3889`/`0.6667` | `1.6529` |
| raw epoch 2 | `0.3450` | `0.5416` | `0.3889`/`0.6667` | `1.5950` |
| epoch-2 `25%` blend | `0.3453` | `0.5432` | `0.3889`/`0.6667` | `1.6389` |

I spent one direct check on the guard-passing blend and added the fresh source
to the exact top-2 good-action book:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability198_probcontext_epoch2_blend_0.25.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 4 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 data/teacher/policyhead192_mat4_latest_fullgame_context_t05_v1 data/teacher/policyhead192_modelblunder_prob_bw1_firstblunders_context_t05_v1 \
  --good-action-book-top-k 2 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_stability198_probcontext_blend025_top2_mat4_stockfish_gate.pgn
```

Result: `0.0/2`. Exact coverage did change both repeated motifs: the White game
used `Bxe4` instead of the prior `Bc4`, and the Black game used `...c5` instead
of `...Kh8`. Both games still lost in nearby tactical lines, so this remains a
diagnostic source rather than a direct-strength improvement.

## Verification

- `python3 -m compileall -q src/alpha_chess`: passed
- `git diff --check`: passed
- `uv run pytest tests/test_cli.py tests/test_model_blunders.py tests/test_hard_negatives.py -q`:
  `8 passed in 1.63s`
- `uv run pytest tests/test_model_and_data.py tests/test_cli.py -q`:
  `33 passed in 11.81s`
- `uv run pytest tests/test_evaluate.py -q`: `3 passed in 3.82s`
- `uv run pytest -q`: `129 passed in 94.46s`
