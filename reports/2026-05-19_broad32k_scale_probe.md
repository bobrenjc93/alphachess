# Broad32k Stockfish Scale Probe

Timestamp: `2026-05-19T17:07:29-07:00`

## Teacher Data

Generated a broader ignored local Stockfish MultiPV set:

```text
data/teacher/stockfish_multipv_elo1800_32768_t005
```

Config:

- source: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- `engine_time=0.005`
- `multipv=4`
- `policy_temperature_cp=200`
- `min_elo=1800`
- `min_initial_seconds=180`
- `position_stride=2`
- positions: `32768`
- games seen: `22930`
- games used: `883`
- files: `32`

Baseline validation on this new set:

| Checkpoint | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| qvalue parent | `0.3473` | `0.6047` | `0.7142` |
| policy-head broad | `0.3406` | `0.6015` | `0.7154` |
| direct-loss mix parent | `0.3521` | `0.6062` | `0.7208` |

The direct-loss mix parent was the strongest existing checkpoint on this
broader teacher diagnostic.

## Training Run

Run:

```text
experiments/policyhead-broad32k-allloss-directmix-v1
```

Config highlights:

- GPU: A100 reservation `6f1cbec6`
- checkpoint: `experiments/policyhead-lossblunder-directmix-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `epochs=2`
- `lr=0.0000015`
- `bad_action_weight=0.05`
- replay data:
  - broad32k `0.80`
  - `alpha_loss_badactions_all_v1` `0.10`
  - `policyhead_hardneg_lossblunders_v1` `0.10`
- forced Stockfish gate with `promotion_score=0.0`

Results:

| Check | Score |
| --- | ---: |
| parent/internal vs direct-loss mix parent | `0.0/8` |
| forced Stockfish gate | `0.0/2` |

PGN:

```text
reports/policyhead_broad32k_allloss_directmix_stockfish_gate.pgn
```

Validation:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k | `0.3310` | `0.5871` | `0.7047` | N/A |
| `alpha_loss_badactions_all_v1` | `0.1544` | `0.4235` | `0.5666` | `2.2101` |
| `policyhead_hardneg_lossblunders_v1` | `0.0400` | `0.3400` | `0.4500` | `2.6169` |

## Conclusion

Rejected. Simply doubling the broad Stockfish teacher scale and fine-tuning the
policy head from the direct-loss mix parent did not improve robustness. It
regressed the new broad32k validation, collapsed the parent match, and still
failed direct Stockfish.

## Epoch 1 Follow-Up

Timestamp: `2026-05-19T17:18:35-07:00`

The final checkpoint collapsed, but the saved first epoch had a better
validation profile:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k | `0.3563` | `0.6105` | `0.7209` | N/A |
| `alpha_loss_badactions_all_v1` | `0.1856` | `0.4504` | `0.5609` | `2.3612` |
| `policyhead_hardneg_lossblunders_v1` | `0.1300` | `0.3700` | `0.5300` | `2.6850` |

Direct checks on
`experiments/policyhead-broad32k-allloss-directmix-v1/checkpoints/iter_0001/epoch_0001.pt`:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs direct-loss mix parent | `6.0/8` | `reports/policyhead_broad32k_epoch1_vs_directmix_parent.pgn` |
| Stockfish gate | `0.0/2` | `reports/policyhead_broad32k_epoch1_stockfish_gate.pgn` |

Epoch 1 is a better checkpoint than epoch 2 and shows that the larger broad set
can improve fixed diagnostics before overtraining. It still fails direct
Stockfish, so future broad-scale runs need explicit validation/early stopping
and a direct-play gate rather than trusting the last epoch.

## Validation-Selected Follow-Up

Timestamp: `2026-05-19T17:39:49-07:00`

After adding `--select-best-by`, I reran the same broad32k/all-loss/direct-loss
policy-head mix with four epochs and selected `latest.pt` by
`val_source_0_policy_acc` so the broad32k validation source controls the saved
candidate:

```text
experiments/policyhead-broad32k-allloss-directmix-selectbest-v1
```

Config highlights:

- GPU: A100 reservation `79298aab`
- checkpoint: `experiments/policyhead-lossblunder-directmix-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `epochs=4`
- `lr=0.0000015`
- `bad_action_weight=0.05`
- `select_best_by=val_source_0_policy_acc`
- replay weights: broad32k `0.80`, all-loss bad actions `0.10`,
  direct-loss blunders `0.10`

Selection metrics from the internal validation split:

| Epoch | `val_source_0_policy_acc` | Saved as `latest.pt` |
| --- | ---: | --- |
| `1` | `0.3555` | yes |
| `2` | `0.3265` | no |
| `3` | `0.3521` | no |
| `4` | `0.3497` | no |

External validation of the selected `latest.pt`:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k | `0.3551` | `0.6109` | `0.7208` | N/A |
| `alpha_loss_badactions_all_v1` | `0.1898` | `0.4504` | `0.5595` | `2.3581` |
| `policyhead_hardneg_lossblunders_v1` | `0.1300` | `0.3700` | `0.5300` | `2.6701` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs direct-loss mix parent | `4.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/policyhead_broad32k_selectbest_stockfish_gate.pgn` |

The new selector did prevent the final-epoch collapse: `latest.pt` is epoch 1,
not epoch 4, and the broad32k diagnostics match the manually inspected epoch-1
checkpoint. It still failed both Stockfish games, so validation selection is a
process fix rather than a strength breakthrough.

## Root-Guard Check on Selected Checkpoint

Timestamp: `2026-05-19T17:43:06-07:00`

I also ran the selected checkpoint with the current worst-depth root material
guard and root king-safety guard:

| Setting | Direct Stockfish score | PGN |
| --- | ---: | --- |
| `root_material_search_plies=3`, `root_material_max_loss_cp=100`, `root_king_safety_search_plies=2`, `root_king_safety_max_loss_cp=100` | `0.0/2` | `reports/policyhead_broad32k_selectbest_rootguards_stockfish.pgn` |

The guarded losses still collapse tactically, so this checkpoint needs a
stronger policy/value training signal rather than only the current root filters.

## Select-Loss Targeted Repair

Timestamp: `2026-05-19T18:01:50-07:00`

I mined the selected checkpoint's direct Stockfish failures into a new ignored
teacher slice:

```text
data/teacher/selectbest_broad32k_lossblunders_v1
```

Inputs were:

- `reports/policyhead_broad32k_selectbest_stockfish_gate.pgn`
- `reports/policyhead_broad32k_selectbest_rootguards_stockfish.pgn`

Generation settings: `engine_time=0.05`, `multipv=8`,
`policy_temperature_cp=180`, `min_value_delta=0.08`, `position_stride=1`,
`pv_plies=4`, `game_line_plies=2`, `player_name=AlphaChess`.

The dataset has `133` positions and `19` bad-action labels. Baseline validation
on this new slice:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| direct-loss mix parent | `0.2256` | `0.4361` | `0.5414` | `2.5665` |
| selected broad32k parent | `0.2331` | `0.4436` | `0.5338` | `2.5458` |

Repair run:

```text
experiments/policyhead-broad32k-selectloss-repair-v1
```

Config highlights:

- GPU: A100 reservation `920cab41`
- checkpoint: `experiments/policyhead-broad32k-allloss-directmix-selectbest-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `epochs=4`
- `lr=0.000001`
- `bad_action_weight=0.15`
- `select_best_by=val_source_2_bad_action_loss`
- replay weights: broad32k `0.70`, all-loss bad actions `0.15`,
  selected-loss blunders `0.15`

The selector chose epoch 2:

| Epoch | `val_source_2_bad_action_loss` | Saved as `latest.pt` |
| --- | ---: | --- |
| `1` | `1.7405` | yes |
| `2` | `1.6819` | yes |
| `3` | `1.7839` | no |
| `4` | `1.6942` | no |

External validation of the selected repair:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k | `0.3596` | `0.6129` | `0.7224` | N/A |
| `alpha_loss_badactions_all_v1` | `0.2040` | `0.4518` | `0.5637` | `2.3883` |
| `selectbest_broad32k_lossblunders_v1` | `0.2331` | `0.4361` | `0.5489` | `2.4860` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs selected broad32k parent | `2.0/8` | N/A |
| first Stockfish smoke | `0.5/2` | `reports/policyhead_broad32k_selectloss_repair_stockfish_gate.pgn` |
| 4-game Stockfish confirmation | `0.0/4` | `reports/policyhead_broad32k_selectloss_repair_stockfish_confirm.pgn` |

This targeted replay produced another small direct draw and the best broad32k
fixed top-1 so far, but it regressed the parent match and the direct draw did
not confirm over four games. It is not a promotion candidate.

Root-guard confirmation on the same policy-head repair:

| Setting | Direct Stockfish score | PGN |
| --- | ---: | --- |
| `root_material_search_plies=3`, `root_material_max_loss_cp=100`, `root_king_safety_search_plies=2`, `root_king_safety_max_loss_cp=100` | `0.0/4` | `reports/policyhead_broad32k_selectloss_repair_rootguards_stockfish_confirm.pgn` |

The current root guards did not stabilize the unconfirmed draw.

## Full-Network Select-Loss Repair

Timestamp: `2026-05-19T18:14:27-07:00`

I reran the selected-loss repair as a low-learning-rate full-network update
rather than policy-head-only:

```text
experiments/fullnet-broad32k-selectloss-repair-v1
```

Config highlights:

- GPU: A100 reservation `2d03ce86`
- checkpoint: `experiments/policyhead-broad32k-allloss-directmix-selectbest-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=false`
- `epochs=3`
- `lr=0.0000005`
- `value_weight=0.05`
- `bad_action_weight=0.15`
- `select_best_by=val_source_2_bad_action_loss`
- replay weights: broad32k `0.75`, all-loss bad actions `0.10`,
  selected-loss blunders `0.15`

The selector chose epoch 3:

| Epoch | `val_source_2_bad_action_loss` | Saved as `latest.pt` |
| --- | ---: | --- |
| `1` | `1.9683` | yes |
| `2` | `1.9225` | yes |
| `3` | `1.8879` | yes |

External validation of the selected checkpoint:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k | `0.3383` | `0.5987` | `0.7145` | N/A |
| `alpha_loss_badactions_all_v1` | `0.1742` | `0.4405` | `0.5751` | `2.2300` |
| `selectbest_broad32k_lossblunders_v1` | `0.2180` | `0.4436` | `0.5263` | `2.3502` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs selected broad32k parent | `2.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/fullnet_broad32k_selectloss_repair_stockfish_gate.pgn` |

Unfreezing the trunk reduced bad-action loss on the all-loss and selected-loss
slices, but it regressed broad32k policy accuracy and failed both direct
Stockfish games. The policy-head-only targeted repair remains a better
diagnostic checkpoint, though neither branch confirmed direct strength.

## Broad32k Hard-Label Selector Repair

Timestamp: `2026-05-19T18:48:42-07:00`

I retried the selected broad32k parent with hard action labels preferred for
the broad32k source, while still mixing in all-loss and selected-loss
bad-action data:

```text
experiments/policyhead-broad32k-hardlabels-selectbest-v1
```

Config highlights:

- GPU: A100 reservation `8c2008a1`
- checkpoint: `experiments/policyhead-broad32k-allloss-directmix-selectbest-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `prefer_action_labels=true`
- `epochs=3`
- `lr=0.0000008`
- `bad_action_weight=0.10`
- `select_best_by=val_source_0_policy_acc`
- replay weights: broad32k `0.80`, all-loss bad actions `0.10`,
  selected-loss blunders `0.10`

The selector chose epoch 3:

| Epoch | `val_source_0_policy_acc` | Saved as `latest.pt` |
| --- | ---: | --- |
| `1` | `0.3410` | yes |
| `2` | `0.3602` | yes |
| `3` | `0.3605` | yes |

Selected checkpoint validation:

| Source | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad32k hard-label split | `0.3605` | `0.6136` | `0.7228` | N/A |
| `alpha_loss_badactions_all_v1` split | `0.1538` | `0.4615` | `0.5077` | `2.5569` |
| `selectbest_broad32k_lossblunders_v1` split | `0.1053` | `0.3684` | `0.4737` | `1.7510` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs selected broad32k parent | `6.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/policyhead_broad32k_hardlabels_selectbest_stockfish_gate.pgn` |

Hard-label broad32k tuning produced the best broad32k source-0 top-1 so far
and beat the selected broad32k parent internally, but both direct Stockfish
games were losses. This remains diagnostic progress rather than a promotion
candidate.
