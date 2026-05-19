# Hard-Negative Policy Repair

Timestamp: `2026-05-19T13:57:45-07:00`

## Replay Mining

Added `alpha-chess hard-negatives`, which mines the checkpoint's top legal wrong
move as `bad_actions` for Stockfish-labeled replay.

Mined local ignored data:

```text
data/teacher/policyhead16k_hardneg_v1
```

Summary:

```text
checkpoint=experiments/policyhead-16k-leafloss-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt
data=['data/teacher/stockfish_multipv_elo1800_16384', 'data/teacher/alpha_loss_leafmcts_v1']
positions=16958
hard_negative_positions=10015
top1_error_rate=0.590577
chunks=17
```

## Training

Run:

```text
experiments/policyhead-hardneg16k-v1
```

Config highlights:

- GPU: A100 reservation `327cd3bb`
- Base checkpoint: `experiments/policyhead-16k-leafloss-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`
- `games=0`
- `epochs=3`
- `batch_size=256`
- `lr=0.000005`
- `value_weight=0.0`
- `bad_action_weight=0.25`
- `bad_action_margin=1.0`
- `legal_policy_loss=true`
- `prefer_action_labels=true`
- `policy_head_only=true`
- replay data: `data/teacher/policyhead16k_hardneg_v1`

Parent match:

```text
score=2.0/8
wins=0
draws=4
losses=4
promoted=false
```

## Diagnostics

Validation on the mined replay with `bad_action_weight=0.25`:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| base policy-head 16k | `0.4094` | `0.6681` | `0.7721` | `3.2213` |
| hard-negative repair | `0.3927` | `0.6555` | `0.7678` | `2.8387` |

The repair reduced the bad-action margin loss, but it also hurt target top-k
accuracy enough to lose the parent match.

## Direct Stockfish Checks

Local Stockfish, `engine_time=0.05`, `64` simulations,
`material_value_weight=0.15`, `material_value_search_plies=2`, `max_plies=180`.

| Probe | Score | PGN |
| --- | ---: | --- |
| first smoke | `0.5/2` | `reports/policyhead_hardneg16k_stockfish_64sims.pgn` |
| confirmation | `0.0/4` | `reports/policyhead_hardneg16k_stockfish_64sims_confirm.pgn` |

The first smoke reproduced a direct Stockfish draw, but the larger confirmation
did not hold. Hard-negative mining is promising as a diagnostic/training tool,
but the current weight/regime over-penalizes wrong top moves and reduces broad
policy accuracy.

## Lower-Pressure Follow-Up

Timestamp: `2026-05-19T14:10:09-07:00`

Run:

```text
experiments/policyhead-hardneg16k-bw005-v1
```

Config changes from the first hard-negative run:

- GPU: A100 reservation `5c1bc136`
- `epochs=2`
- `lr=0.000003`
- `bad_action_weight=0.05`

Parent match:

```text
score=6.0/8
wins=4
draws=4
losses=0
```

Stockfish gate:

```text
score=0.0/2
wins=0
draws=0
losses=2
pgn=reports/policyhead_hardneg16k_bw005_stockfish_gate.pgn
```

Validation on the mined replay:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| lower-pressure repair | `0.3923` | `0.6533` | `0.7674` | `2.8639` |

The lower-pressure setting preserved internal parent strength but still reduced
top-k accuracy relative to the base and failed the direct Stockfish gate. The
next hard-negative attempt should mix in an explicit unpenalized broad-policy
source or use a smaller subset of the mined negatives instead of applying the
margin to every top-1 error.

## Mixed Broad-Label Follow-Up

Timestamp: `2026-05-19T14:26:11-07:00`

Run:

```text
experiments/policyhead-hardneg16k-mix-bw005-v1
```

Config changes from the lower-pressure run:

- GPU: A100 reservation `3c9cb7ff`
- `lr=0.000002`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_16384`, weight `0.45`
  - `data/teacher/alpha_loss_leafmcts_v1`, weight `0.10`
  - `data/teacher/policyhead16k_hardneg_v1`, weight `0.45`

Parent match:

```text
score=8.0/8
wins=8
draws=0
losses=0
```

Stockfish checks:

```text
gate score=0.0/2
gate pgn=reports/policyhead_hardneg16k_mix_bw005_stockfish_gate.pgn
confirmation score=0.0/4
confirmation pgn=reports/policyhead_hardneg16k_mix_bw005_stockfish_confirm.pgn
```

Validation on the mined replay:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| mixed broad-label repair | `0.3937` | `0.6544` | `0.7677` | `2.8593` |

This was the strongest hard-negative parent match so far and preserved top-k
slightly better than the single-source hard-negative repairs. It still failed
both direct Stockfish checks, so internal policy/search gains remain a poor
proxy for direct strength.

### Mixed Checkpoint Root-Filter Checks

Timestamp: `2026-05-19T14:30:59-07:00`

Follow-up direct checks on
`experiments/policyhead-hardneg16k-mix-bw005-v1/checkpoints/iter_0001/latest.pt`
also failed:

| Variant | Direct Stockfish score | PGN |
| --- | ---: | --- |
| `root_material_search_plies=3`, `root_material_max_loss_cp=100` | `0.0/2` | `reports/policyhead_hardneg16k_mix_rootmaterial100_stockfish.pgn` |
| `root_king_safety_search_plies=2`, `root_king_safety_max_loss_cp=100` | `0.0/2` | `reports/policyhead_hardneg16k_mix_rootking100_stockfish.pgn` |

The existing tactical root filters did not convert the mixed hard-negative
checkpoint's internal gain into direct Stockfish strength.

## Soft Broad-Policy Follow-Up

Timestamp: `2026-05-19T14:51:35-07:00`

Run:

```text
experiments/policyhead-hardneg16k-mixsoft-bw005-v1
```

This repeated the mixed broad-label run but left `prefer_action_labels=false`,
so `data/teacher/stockfish_multipv_elo1800_16384` contributed its soft MultiPV
policy target instead of a single hard action. The mined hard-negative and
latest-loss sources still supplied action labels and `bad_actions`.

Config highlights:

- GPU: A100 reservation `f6c4bd2f`
- `epochs=2`
- `lr=0.000002`
- `bad_action_weight=0.05`
- `policy_head_only=true`
- replay weights: `0.45 0.10 0.45`

Parent match:

```text
score=6.0/8
wins=4
draws=4
losses=0
```

Stockfish gate:

```text
score=0.0/2
pgn=reports/policyhead_hardneg16k_mixsoft_bw005_stockfish_gate.pgn
```

Validation on the mined replay:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| soft broad-policy repair | `0.3925` | `0.6541` | `0.7670` | `2.8512` |

Keeping the broad Stockfish policy soft did not materially improve the hard
target top-k metrics or the direct Stockfish gate.

## Direct Loss-Blunder Replay Follow-Up

Timestamp: `2026-05-19T15:34:00-07:00`

The latest soft broad-policy direct losses included high-confidence policy
top-1 blunders, so this follow-up generated a focused bad-action replay from
recent hard-negative policy-head Stockfish loss PGNs:

```text
data/teacher/policyhead_hardneg_lossblunders_v1
```

Teacher generation:

- sources: 9 recent `policyhead_hardneg16k*` Stockfish-loss PGNs
- games seen/used: `22`
- positions: `100`
- bad-action positions: `100`
- value-delta range: `0.0830` to `0.4903`; mean `0.1577`
- Stockfish: `engine_time=0.05`, `multipv=8`, `policy_temperature_cp=180`

Two policy-head-only repair runs started from
`experiments/policyhead-hardneg16k-mixsoft-bw005-v1/checkpoints/iter_0001/latest.pt`:

| Run | GPU | Replay mix | Parent/internal score | Direct Stockfish score | PGN |
| --- | --- | --- | ---: | ---: | --- |
| `experiments/policyhead-hardneg-lossblunder-mixsoft-v1` | A100 `7bd32576` | broad 16k `0.55`, mined hardneg `0.25`, direct-loss blunders `0.20`; `epochs=3`, `lr=0.0000015`, `bad_action_weight=0.10` | `4.0/8` vs soft-mix parent | `0.0/2` | `reports/policyhead_hardneg_lossblunder_mixsoft_stockfish_gate.pgn` |
| `experiments/policyhead-lossblunder-directmix-v1` | A100 `9f1a32c9` | broad 16k `0.65`, direct-loss blunders `0.35`; `epochs=4`, `lr=0.000002`, `bad_action_weight=0.20`; forced gate with `promotion_score=0.0` | `8.0/8` vs soft-mix parent | `0.0/2` | `reports/policyhead_lossblunder_directmix_stockfish_gate.pgn` |
| `experiments/fullnet-lossblunder-directmix-v1` | A100 `8ecd9699` | full-network follow-up from the direct-loss mix; broad 16k `0.65`, direct-loss blunders `0.35`; `epochs=3`, `lr=0.00000075`, `value_weight=0.05`, `bad_action_weight=0.20`; forced gate with `promotion_score=0.0` | `4.0/8` vs direct-loss mix parent | `0.0/2` | `reports/fullnet_lossblunder_directmix_stockfish_gate.pgn` |

Validation on `data/teacher/policyhead_hardneg_lossblunders_v1`:

| Checkpoint | Top-1 | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| soft broad-policy parent | `0.1200` | `0.3000` | `0.4600` | `2.8098` |
| balanced loss-blunder repair | `0.1200` | `0.3500` | `0.5000` | `2.7250` |
| direct-loss-only mix repair | `0.1200` | `0.3500` | `0.5000` | `2.6989` |
| full-network direct-loss repair | `0.1300` | `0.3300` | `0.5000` | `2.4785` |

The targeted replay moved top-3/top-5 and margin loss slightly, and the more
aggressive direct-loss mix dominated the soft-mix parent internally. Unfreezing
the trunk with a very low learning rate reduced bad-action loss further and
nudged target top-1 to `0.1300`, but it also regressed broad 16k Stockfish
teacher accuracy to `0.3762` top-1 / `0.6453` top-3 / `0.7579` top-5 and still
failed the direct Stockfish gate. The current direct-loss replay is too narrow
or too weak to repair high-confidence direct-play blunders by fine-tuning alone.
