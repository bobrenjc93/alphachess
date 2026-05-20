# Stockfish-Confirmed Model-Blunder Mining

Timestamp: `2026-05-20T00:22:48-07:00`

## Summary

The previous hard-negative runs penalized moves that disagreed with the teacher
target. This probe adds a sharper miner: take the model's top legal move, and
only store it as a bad action when Stockfish confirms that playing it drops the
root value by at least a configured threshold.

The mined data is a better diagnostic signal than label disagreement alone:
`2,010` confirmed blunders were found in `16,384` broad/t05 teacher positions.
However, a small policy-head-only repair reduced the confirmed-blunder loss only
slightly and still scored `0.0/2` against Stockfish.

## Code change

New CLI:

```bash
uv run alpha-chess model-blunders \
  --checkpoint CHECKPOINT \
  --data TEACHER_DIR... \
  --out OUT_DIR \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.02 \
  --max-positions 16384 \
  --min-value-delta 0.08 \
  --bad-actions-per-position 1 \
  --prefer-action-labels
```

The miner writes replay NPZ files containing the teacher action as `actions`,
the model move as `bad_actions`, and a padded `value_deltas` array for the
Stockfish-confirmed drops.

Verification:

```bash
uv run pytest
```

Result: `94 passed`.

## Mined data

- Checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Data:
  `data/teacher/stockfish_multipv_elo1800_8192_t05`,
  `data/teacher/stockfish_multipv_elo1800_65536_t005`
- Output: `data/teacher/policyhead192_stockfish_confirmed_blunders_v1`
- Engine: `tools/stockfish/bin/stockfish`
- Engine time: `0.02`
- Max positions: `16384`
- Min value delta: `0.08`

Summary:

| Metric | Value |
| --- | ---: |
| positions seen | `16384` |
| model-wrong positions | `4540` |
| Stockfish-confirmed blunder positions | `2010` |
| bad actions | `2010` |
| value-delta min | `0.0802` |
| value-delta mean | `0.2725` |
| value-delta max | `1.9042` |

Baseline validation of the starting checkpoint:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| confirmed blunders | `0.0000` | `0.3343` | `0.5005` | `4.0346` |
| disjoint holdout | `0.3442` | `0.5459` | `0.6464` | N/A |

## Repair run

- Experiment:
  `experiments/policyhead192-stockfish-confirmed-blunder-repair-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Training data weights:
  broad65k `0.43`, t05 `0.24`, confirmed blunders `0.23`,
  fullnet loss slice `0.05`, latest loss slice `0.05`
- Epochs: `3`
- LR: `4e-6`
- Bad-action weight: `0.20`
- Selection metric: `holdout_policy_acc`
- Selected epoch: `1`

Selected checkpoint metrics:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| disjoint holdout | `0.3438` | `0.5459` | `0.6472` | N/A |
| confirmed blunders | `0.0000` | `0.3803` | `0.5070` | `3.9276` |

Direct gate:

| Check | Result | Artifact |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_stockfish_confirmed_blunder_repair_gate.pgn` |

## H100 full-network follow-ups

The policy-head-only repair did not move target top-1, so I used the active
H100 reservation to try two trunk-unfrozen repairs from the same parent.

| Run | Selection | Key settings | Disjoint holdout | Confirmed blunders | Direct gate |
| --- | --- | --- | ---: | ---: | ---: |
| `experiments/policyhead192-stockfish-confirmed-fullnet-v1/checkpoints/iter_0001` | best `holdout_policy_acc`, epoch `3` | LR `1e-6`, bad-action weight `0.20`, weights `0.43/0.24/0.23/0.05/0.05` | top-1 `0.3446`, top-3 `0.5470`, top-5 `0.6483` | top-1 `0.0199`, top-3 `0.3423`, top-5 `0.5075`, bad-action loss `4.0469` | `0.0/2` (`reports/policyhead192_stockfish_confirmed_fullnet_gate.pgn`) |
| `experiments/policyhead192-stockfish-confirmed-fullnet-badmargin-v1/checkpoints/iter_0001` | best confirmed-blunder bad-action loss, epoch `5` | LR `7.5e-7`, bad-action weight `0.50`, weights `0.40/0.22/0.28/0.05/0.05` | top-1 `0.3439`, top-3 `0.5470`, top-5 `0.6476` | top-1 `0.0209`, top-3 `0.3458`, top-5 `0.5095`, bad-action loss `4.0162` | `0.0/2` (`reports/policyhead192_stockfish_confirmed_fullnet_badmargin_gate.pgn`) |

## Read

Stockfish-confirmed model-blunder mining is useful because it separates true
value-dropping model choices from harmless label disagreements. The first repair
attempt shows the current policy-head-only recipe is too weak: it moves top-k
and bad-action loss slightly, but target top-1 remains `0.0` on the mined set
and direct play is unchanged.

The H100 full-network follow-ups did get confirmed-blunder target top-1 off
zero, but only to about `2%`. The bad-margin selector also improved the mined
bad-action margin relative to the parent, while slightly regressing the disjoint
holdout. Both still scored `0.0/2` against Stockfish, so this signal is useful
diagnostically but is not yet strong enough as a small replay slice.

Next, this signal should be mined at larger scale or turned into a move
filtering/ranking objective that affects many more direct-game positions.
