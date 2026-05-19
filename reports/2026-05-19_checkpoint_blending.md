# Checkpoint Blending Probe

Date: 2026-05-19

## Change

Added:

```bash
uv run alpha-chess blend-checkpoints \
  --checkpoint-a A.pt \
  --checkpoint-b B.pt \
  --weight-b 0.50 \
  --out blended.pt
```

The command linearly interpolates floating tensors from two checkpoints with the same model config and stores blend metadata in the output checkpoint.

Verification:

- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest tests/test_model_and_data.py`
  - `8 passed`
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest`
  - `56 passed`

## Blend probe

Checkpoint A:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Checkpoint B:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Generated ignored local blends:

- `experiments/blends/qvalue_pvrecent_w025.pt`
- `experiments/blends/qvalue_pvrecent_w050.pt`
- `experiments/blends/qvalue_pvrecent_w075.pt`

## Compact validation

Legal-policy validation on:

- `data/teacher/stockfish_multipv_elo1800_4096`
- `data/teacher/alpha_poisoned_captures_v2`
- `data/teacher/alpha_loss_pvlines_recent_v1`

| Blend weight B | Stockfish 4096 | Poisoned | PV recent |
| ---: | ---: | ---: | ---: |
| `0.25` | `0.5569` | `0.1071` | `0.2044` |
| `0.50` | `0.5557` | `0.2857` | `0.2240` |
| `0.75` | `0.5479` | `0.5000` | `0.2344` |

The blends underperform both source checkpoints on broad Stockfish policy accuracy.

## Direct Stockfish smoke

Tactical-heavy blend:

`experiments/blends/qvalue_pvrecent_w075.pt`

Result:

- 16 simulations, 2 Stockfish games
- score: `0.0/2`
- PGN: `reports/blend_qvalue_pvrecent_w075_vs_stockfish_16sims.pgn`

## Conclusion

Rejected. Simple weight interpolation between the broad qvalue checkpoint and the recent PV-line checkpoint does not preserve the useful parts of either model, likely because batch-norm statistics and fine-tune geometry do not align cleanly.
