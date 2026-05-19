# Policy-Head-Only Hard-Label Probe

Date: 2026-05-19

## Change Under Test

Added `prefer_action_labels` support so training and validation can use sparse
best-move `actions` even when a teacher file also contains dense MultiPV
`policies`. This allows hard best-move supervision from existing Stockfish
teacher files without regenerating data.

Implementation:

- `SelfPlayDataset(..., prefer_action_labels=True)` suppresses `policy` only
  when `actions` are present.
- `alpha-chess train`, `validate`, and `iterate` accept
  `--prefer-action-labels`.
- `scripts/submit_gpu_iteration.sh` accepts `PREFER_ACTION_LABELS=1`.

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_model_and_data.py tests/test_iteration.py
24 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
74 passed
```

## Run

`experiments/policyhead-hardlabels-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `policy_head_only=true`
- `prefer_action_labels=true`
- `value_weight=0.0`
- `lr=1e-5`
- `games=0`
- `legal_policy_loss=true`
- `max_plies=180`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v4`
  - `data/teacher/alpha_loss_pvlines_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.60 0.20 0.08 0.10 0.02`

GPU runner:

- `gpu-dev` A100 reservation `8ad3fc97`
- completed and copied back normally

Candidate:

`experiments/policyhead-hardlabels-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Training checkpoint metrics:

```text
epoch_loss=1.9291
val_loss=1.9963
val_policy_loss=1.9963
val_policy_acc=0.4562
```

## Promotion

Parent match against qvalue:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/4`
- wins/draws/losses: `0/0/4`
- PGN: `reports/policyhead_hardlabels_qvalue_stockfish_gate_16sims.pgn`
- promoted: `false`

Additional 16-simulation inference sweep:

| Setting | Games | Score | PGN |
| --- | ---: | ---: | --- |
| `root_material_search_plies=2`, `root_material_max_loss_cp=250` | 2 | `0.0/2` | `reports/policyhead_hardlabels_qvalue_rootmaterial250_stockfish_16sims.pgn` |
| `material_value_weight=0.30` | 2 | `0.0/2` | `reports/policyhead_hardlabels_qvalue_material030_stockfish_16sims.pgn` |
| `c_puct=0.8` | 2 | `0.0/2` | `reports/policyhead_hardlabels_qvalue_cpuct08_stockfish_16sims.pgn` |
| `policy_prior_temperature=2.0` | 2 | `0.0/2` | `reports/policyhead_hardlabels_qvalue_priorT2_stockfish_16sims.pgn` |

## Fixed Validation

Candidate-only CPU validation used batch size 512, legal policy loss,
`prefer_action_labels=true`, and `value_weight=0.25`.

| Dataset | Examples | Policy Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096_t005` | 4096 | `0.5903` |
| `stockfish_multipv_elo1800_4096` | 4096 | `0.6147` |
| `puzzles/lines_1200_2400_100k` | 100000 | `0.4569` |
| `alpha_loss_reports_v4` | 222 | `0.4775` |
| `alpha_loss_pvlines_v1` | 512 | `0.2520` |
| `alpha_poisoned_captures_v2` | 28 | `0.1429` |

Overall policy accuracy was `0.4669` over 108954 examples.

## Conclusion

Rejected as a direct-play candidate. Hard best-move labels improved the parent
match and fixed Stockfish/puzzle/action-label diagnostics relative to the soft
v4/PV policy-head run, but the direct Stockfish gate remained `0.0/4`, and the
small inference sweep did not recover a draw. The losses still show short
tactical collapses and mating attacks, so hard labels alone are not sufficient.
