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
