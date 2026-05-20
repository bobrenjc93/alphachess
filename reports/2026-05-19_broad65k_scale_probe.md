# Broad65k Stockfish Scale Probe

Timestamp: `2026-05-19T19:34:49-07:00`

## Teacher Data

Generated a doubled ignored local Stockfish MultiPV set:

```text
data/teacher/stockfish_multipv_elo1800_65536_t005
```

Config:

- source: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- `engine_time=0.005`
- `multipv=4`
- `policy_temperature_cp=200`
- `min_elo=1800`
- `min_initial_seconds=180`
- `position_stride=2`
- positions: `65536`
- games seen: `46099`
- games used: `1734`
- files: `64`

Baseline validation on this new set:

| Checkpoint | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| hard-label broad32k parent | `0.3433` | `0.5850` | `0.6971` |
| hard-label loss-repair parent | `0.3486` | `0.5897` | `0.7011` |

The hard-label loss-repair checkpoint remained stronger on the larger
diagnostic before additional 65k training.

## Training Run

Run:

```text
experiments/policyhead-broad65k-hardlabel-selectbest-v1
```

Config highlights:

- GPU: A100 reservation `2c1d9530`
- checkpoint: `experiments/policyhead-broad32k-hardlabel-lossrepair-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `prefer_action_labels=true`
- `epochs=3`
- `lr=0.0000005`
- `bad_action_weight=0.10`
- `select_best_by=val_source_0_policy_acc`
- replay weights: broad65k `0.75`, all-loss bad actions `0.08`,
  selected-loss blunders `0.07`, hard-label loss blunders `0.10`

The selector chose epoch 1:

| Epoch | `val_source_0_policy_acc` | Saved as `latest.pt` |
| --- | ---: | --- |
| `1` | `0.3513` | yes |
| `2` | `0.3511` | no |
| `3` | `0.3429` | no |

External validation of the selected checkpoint:

| Dataset | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| broad65k hard labels | `0.3488` | `0.5899` | `0.7017` | N/A |
| broad32k hard labels | `0.3665` | `0.6194` | `0.7284` | N/A |
| `alpha_loss_badactions_all_v1` | `0.2110` | `0.4533` | `0.5567` | `2.3619` |
| `hardlabel_broad32k_lossblunders_v1` | `0.2597` | `0.5000` | `0.6169` | `2.6493` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs hard-label loss-repair parent | `4.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/policyhead_broad65k_hardlabel_selectbest_stockfish_gate.pgn` |

Doubling the cheap broad teacher set nudged fixed validation metrics but
regressed the parent match and did not move direct Stockfish. The next attempt
should not assume more of the same shallow broad labels will close the direct
tactical gap.

## Expert-Mix Follow-Up

Timestamp: `2026-05-19T19:56:50-07:00`

I mixed the 65k broad Stockfish source with the larger rapid expert PGN import
to test whether human opening coverage would stabilize the policy:

```text
experiments/policyhead-broad65k-expertmix-v1
```

Config highlights:

- GPU: A100 reservation `4307c053`
- checkpoint: `experiments/policyhead-broad32k-hardlabel-lossrepair-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `prefer_action_labels=true`
- `epochs=3`
- `lr=0.0000005`
- `bad_action_weight=0.10`
- `select_best_by=val_source_0_policy_acc`
- replay weights: broad65k `0.55`, rapid expert `0.25`,
  all-loss bad actions `0.10`, hard-label loss blunders `0.10`

The selector chose epoch 3:

| Epoch | `val_source_0_policy_acc` | `val_source_1_policy_acc` | Saved as `latest.pt` |
| --- | ---: | ---: | --- |
| `1` | `0.3387` | `0.4160` | yes |
| `2` | `0.3380` | `0.4135` | no |
| `3` | `0.3407` | `0.4193` | yes |

External validation of the selected checkpoint:

| Dataset | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| broad65k hard labels | `0.3476` | `0.5882` | `0.7012` |
| rapid expert import | `0.4171` | `0.6809` | `0.7919` |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs hard-label loss-repair parent | `4.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/policyhead_broad65k_expertmix_stockfish_gate.pgn` |

The expert mix did improve expert-move validation, but it reduced broad
Stockfish validation and did not improve either the parent match or the direct
Stockfish gate. Expert opening coverage alone is not enough in this mix.
