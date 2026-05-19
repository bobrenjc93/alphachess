# Policy-Head-Only V4/PV Replay Probe

Date: 2026-05-19

## Run

`experiments/policyhead-v4pv-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `policy_head_only=true`
- `value_weight=0.0`
- `lr=1e-5`
- `games=0`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v4`
  - `data/teacher/alpha_loss_pvlines_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.60 0.20 0.08 0.10 0.02`

The GPU wrapper was started on A100 reservation `1e8938eb`. Training completed
and produced the checkpoint, but the wrapper was stopped during the default
512-ply parent match to avoid burning the reservation on an overlong eval. The
checkpoint was copied back and evaluated separately with 180-ply direct games.

Candidate:

`experiments/policyhead-v4pv-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Training checkpoint metrics:

```text
epoch_loss=2.4284
val_loss=2.0110
val_policy_loss=2.0110
val_policy_acc=0.4485
```

## Direct Stockfish Gate

Settings:

- `games=4`
- `simulations=16`
- `engine_time=0.05`
- `max_plies=180`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- CPU evaluation

Result:

| Games | Score | W/D/L | PGN |
| ---: | ---: | --- | --- |
| 4 | `0.0/4` | `0/0/4` | `reports/policyhead_v4pv_qvalue_stockfish_16sims.pgn` |

The games still lose through material drops and king attacks from both colors.

## Fixed Validation

Candidate-only CPU validation used batch size 512, legal policy loss, and
`value_weight=0.25`.

| Dataset | Examples | Policy Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096_t005` | 4096 | `0.4690` |
| `stockfish_multipv_elo1800_4096` | 4096 | `0.5476` |
| `puzzles/lines_1200_2400_100k` | 100000 | `0.4521` |
| `alpha_loss_reports_v4` | 222 | `0.3468` |
| `alpha_loss_pvlines_v1` | 512 | `0.2305` |
| `alpha_poisoned_captures_v2` | 28 | `0.0714` |

Overall policy accuracy was `0.4549` over 108954 examples.

## Conclusion

Rejected. Replacing the broad policy-head source mix with stronger Stockfish
labels plus v4/PV failure-line replay did not improve direct play. It also
regressed broad Stockfish fixed accuracy relative to the qvalue parent, so this
mix is not a better policy-head repair path.
