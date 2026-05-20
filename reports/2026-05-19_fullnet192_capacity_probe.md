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
