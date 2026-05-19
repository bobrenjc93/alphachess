# Color-Mirror Replay Augmentation

Date: 2026-05-19

## Change

Added optional exact color-mirror augmentation for FEN-backed replay:

- `color_mirror_board`, `color_mirror_move`, `color_mirror_action`, and
  `color_mirror_policy` map positions and labels through `python-chess`
  `Board.mirror()`.
- `SelfPlayDataset(..., color_mirror_augmentation=True)` doubles datasets with
  mirrored samples and duplicate source weights.
- `train` and `iterate` accept `--color-mirror-augmentation`.
- `scripts/submit_gpu_iteration.sh` forwards `COLOR_MIRROR_AUGMENTATION=1`.

Verification:

```text
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest
63 passed
```

## Probe

Run:

`experiments/focus-pvrecent-mirroraug-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `lr=2e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `color_mirror_augmentation=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/all_1200_2400_50k`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.50 0.12 0.18 0.04 0.10 0.06`

## Result

Candidate:

`experiments/focus-pvrecent-mirroraug-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/8`
- wins/draws/losses: `0/0/8`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

Fixed validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5193` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3308` |
| `alpha_loss_reports_v2` | `0.6393` | `0.4754` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4490` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6071` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2448` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2715` |

## Conclusion

Keep the augmentation feature, but reject this training setting. A direct
one-epoch low-LR color-mirror fine-tune from PV-recent badly failed the parent
gate and regressed most fixed diagnostics. If reused, mirror augmentation should
be introduced during earlier pretraining or with a much smaller effective update.
