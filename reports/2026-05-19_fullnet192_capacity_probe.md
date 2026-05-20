# Fullnet192 Capacity Probe

Timestamp: `2026-05-19T20:49:36-07:00`

## Full-Network Scratch Run

I trained a larger `192`-channel, `8`-block network from scratch on the broad65k
Stockfish teacher, rapid expert games, and the loss slices:

```text
experiments/fullnet192-broad65k-expertmix-scratch-v1
```

Config highlights:

- GPU: A100 reservation `ec00d78d`
- checkpoint: none, trained from scratch
- channels: `192`
- blocks: `8`
- `epochs=5`
- `batch_size=256`
- `lr=0.001`
- `value_weight=0.10`
- `bad_action_weight=0.05`
- `select_best_by=val_source_0_policy_acc`
- replay weights: broad65k `0.60`, rapid expert `0.25`,
  all-loss bad actions `0.08`, hard-label loss blunders `0.07`

The selector chose epoch 5:

| Epoch | `val_source_0_policy_acc` | `val_source_0_value_loss` | `val_source_3_bad_action_loss` | Saved as `latest.pt` |
| --- | ---: | ---: | ---: | --- |
| `1` | `0.3229` | `0.1045` | `0.0389` | yes |
| `2` | `0.3282` | `0.0636` | `0.0238` | yes |
| `3` | `0.3500` | `0.0591` | `0.0007` | yes |
| `4` | `0.3632` | `0.0567` | `0.0001` | yes |
| `5` | `0.3682` | `0.0585` | `0.0001` | yes |

Fixed validation of the selected checkpoint:

| Dataset | Top-1 | Top-3 | Top-5 | Value loss | Bad-action loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad65k hard labels | `0.8620` | `0.9500` | `0.9674` | `0.0401` | N/A |
| broad32k hard labels | `0.5514` | `0.7244` | `0.7945` | `0.0408` | N/A |
| `hardlabel_broad32k_lossblunders_v1` | `0.9091` | `0.9545` | `0.9610` | `0.0084` | `0.0017` |
| puzzle lines 100k | `0.3224` | `0.5382` | `0.6371` | `1.0810` | N/A |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| vs uniform | `8.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/fullnet192_broad65k_expertmix_scratch_stockfish_gate.pgn` |
| 64 simulations vs Stockfish | `0.0/2` | `reports/fullnet192_broad65k_expertmix_scratch_64sims_stockfish.pgn` |
| policy-only vs Stockfish | `0.0/2` | `reports/fullnet192_broad65k_expertmix_scratch_policyonly_stockfish.pgn` |

The larger model has enough capacity to memorize the broad and small loss
datasets, but that does not transfer to direct Stockfish play. It is also weak
on the held-out tactical puzzle-line source, so the next follow-up targeted
that gap directly.

## Puzzle-Mix Policy-Head Follow-Up

I fine-tuned only the policy head of the 192x8 scratch checkpoint with the
100k puzzle-line source mixed in:

```text
experiments/policyhead192-broad65k-puzzlemix-v1
```

Config highlights:

- GPU: A100 reservation `1573292c`
- checkpoint: `experiments/fullnet192-broad65k-expertmix-scratch-v1/checkpoints/iter_0001/latest.pt`
- `policy_head_only=true`
- `epochs=3`
- `lr=0.00005`
- `bad_action_weight=0.05`
- `select_best_by=val_source_1_policy_acc`
- replay weights: broad65k `0.55`, puzzle lines `0.30`,
  all-loss bad actions `0.08`, hard-label loss blunders `0.07`

The selector chose epoch 3:

| Epoch | `val_source_0_policy_acc` | `val_source_1_policy_acc` | `val_source_2_bad_action_loss` | Saved as `latest.pt` |
| --- | ---: | ---: | ---: | --- |
| `1` | `0.8595` | `0.3317` | `0.2814` | yes |
| `2` | `0.8612` | `0.3398` | `0.2625` | yes |
| `3` | `0.8648` | `0.3442` | `0.2566` | yes |

Direct checks:

| Check | Score | PGN |
| --- | ---: | --- |
| parent/internal vs fullnet192 scratch parent | `2.0/8` | N/A |
| Stockfish gate | `0.0/2` | `reports/policyhead192_broad65k_puzzlemix_stockfish_gate.pgn` |

The puzzle mix improved the puzzle-source validation only modestly and regressed
the parent match. It still failed direct Stockfish, so this mix is not a
promotion path.

## Fullnet192 Loss-Blunder Replay

Timestamp: `2026-05-19T21:18:15-07:00`

I mined the fullnet192 direct losses into another ignored hard-negative replay:

```text
data/teacher/fullnet192_lossblunders_v1
```

Generation inputs:

- `reports/fullnet192_broad65k_expertmix_scratch_stockfish_gate.pgn`
- `reports/fullnet192_broad65k_expertmix_scratch_64sims_stockfish.pgn`
- `reports/fullnet192_broad65k_expertmix_scratch_policyonly_stockfish.pgn`
- `reports/policyhead192_broad65k_puzzlemix_stockfish_gate.pgn`

Generation settings: `engine_time=0.05`, `multipv=8`,
`policy_temperature_cp=180`, `min_value_delta=0.08`, `position_stride=1`,
`pv_plies=4`, `game_line_plies=2`, `player_name=AlphaChess`.

The dataset has `238` positions and `34` bad-action labels. Baseline validation
on this new slice:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| fullnet192 scratch | `0.2059` | `0.4412` | `0.5378` | `4.4038` |
| policyhead192 puzzle mix | `0.2353` | `0.4454` | `0.5294` | `3.6894` |

I attempted to start a targeted policy-head repair:

```text
experiments/policyhead192-fullnetloss-repair-v1
```

Planned config highlights:

- checkpoint: `experiments/fullnet192-broad65k-expertmix-scratch-v1/checkpoints/iter_0001/latest.pt`
- replay weights: broad65k `0.50`, puzzle lines `0.20`,
  fullnet192 loss blunders `0.15`, all-loss bad actions `0.08`,
  hard-label loss blunders `0.07`
- `policy_head_only=true`
- `bad_action_weight=0.15`
- `lr=0.00001`
- `select_best_by=val_source_2_bad_action_loss`

The A100 reservation `26ab7495` never became active because `gpu-dev` reported
`Waiting for disk snapshot to complete (from previous session)`. I canceled the
reservation. This repair is ready to retry once GPU reservations are healthy.

## CPU Fullnet192 Loss Overfit Smoke

Timestamp: `2026-05-19T21:22:07-07:00`

While GPU reservations were blocked, I ran a narrow CPU policy-head-only overfit
on only `fullnet192_lossblunders_v1`:

```text
experiments/policyhead192-fullnetloss-cpu-overfit-v1
```

Config highlights:

- checkpoint: `experiments/fullnet192-broad65k-expertmix-scratch-v1/checkpoints/iter_0001/latest.pt`
- data: `data/teacher/fullnet192_lossblunders_v1`
- `policy_head_only=true`
- `epochs=10`
- `lr=0.00001`
- `bad_action_weight=0.20`
- `select_best_by=val_bad_action_loss`

The selector chose epoch 2 with held-out `val_bad_action_loss=3.8579`.
Full-slice validation worsened versus the parent baseline:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| fullnet192 scratch parent | `0.2059` | `0.4412` | `0.5378` | `4.4038` |
| CPU loss overfit | `0.2059` | `0.4370` | `0.5336` | `4.4458` |

Direct check:

| Check | Score | PGN |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_fullnetloss_cpu_overfit_stockfish_gate.pgn` |

The tiny CPU overfit is rejected. It did not improve the full loss slice or
direct play.
