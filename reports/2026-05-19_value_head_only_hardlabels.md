# Value-Head-Only Hard-Label Probe

Date: 2026-05-19

## Change Under Test

Added `value_head_only` training:

- freezes stem, residual blocks, and policy head
- trains only `model.value_head`
- keeps frozen modules in eval mode
- rejects simultaneous `policy_head_only` and `value_head_only`
- exposed through `alpha-chess train`, `alpha-chess iterate`, and
  `scripts/submit_gpu_iteration.sh` (`VALUE_HEAD_ONLY=1`)

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_model_and_data.py tests/test_iteration.py
26 passed

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
76 passed
```

## Run

`experiments/valuehead-hardlabels-stockfish-qvalue-material015`

Base checkpoint:

`experiments/policyhead-hardlabels-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Config:

- `value_head_only=true`
- `policy_head_only=false`
- `value_weight=1.0`
- `lr=1e-4`
- `epochs=2`
- `games=0`
- `max_plies=180`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`
  - `data/teacher/alpha_loss_reports_v4`
  - `data/teacher/alpha_loss_pvlines_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.65 0.15 0.15 0.05`

Puzzle-line data was intentionally excluded because its synthetic tactical
values are not a Stockfish value calibration target.

GPU runner:

- `gpu-dev` A100 reservation `860e0f0e`
- completed and copied back normally

Candidate:

`experiments/valuehead-hardlabels-stockfish-qvalue-material015/checkpoints/iter_0001/latest.pt`

Training checkpoint metrics:

```text
epoch_loss=2.8315
val_loss=2.5617
val_policy_loss=2.5307
val_policy_acc=0.5320
val_value_loss=0.0310
```

## Promotion

Parent match against the hard-label policy-head checkpoint:

- score: `6.0/8`
- wins/draws/losses: `4/4/0`

Stockfish gate:

- score: `0.0/4`
- wins/draws/losses: `0/0/4`
- PGN: `reports/valuehead_hardlabels_stockfish_gate_16sims.pgn`
- promoted: `false`

## Conclusion

Rejected as a direct-play candidate. Value-head-only calibration fit the
Stockfish-valued replay data well and preserved internal parent-match strength,
but it still failed the direct Stockfish gate. The direct losses remain short
tactical collapses, including queen and king-safety failures, so value-head
recalibration alone is not sufficient.
