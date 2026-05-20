# Engine Self-Play Trajectory Probe

Timestamp: `2026-05-20T10:53:16-07:00`

## Summary

I added a compact Stockfish-vs-Stockfish trajectory source using the new
`engine-self-play` command and tested whether it helps the current best
broad-holdout fullnet checkpoint. It did not. The initial selected policy-head
tune lowered policy losses but regressed top-1 accuracy on every validation
source. A lower-weight ablation reduced the loss further and improved recent120
v2 top-1, but it still regressed broad holdout and engine-trajectory top-1, so
both variants were rejected before a direct Stockfish gate.

## Data

Generated engine games:

```bash
uv run alpha-chess engine-self-play \
  --out data/engine_selfplay/stockfish_selfplay_96x64_t01_v1.pgn \
  --engine-path tools/stockfish/bin/stockfish \
  --games 96 \
  --max-plies 64 \
  --engine-time 0.01 \
  --opening-random-plies 12 \
  --opening-multipv 6 \
  --opening-temperature-cp 80 \
  --seed 120
```

Result: `96` games, average `63.96` plies, result counts `{'0-1': 1,
'1/2-1/2': 95}`.

Generated teacher slice:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn data/engine_selfplay/stockfish_selfplay_96x64_t01_v1.pgn \
  --out data/teacher/stockfish_selfplay_4096_t01_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --max-positions 4096 \
  --position-stride 1 \
  --multipv 6 \
  --policy-temperature-cp 180 \
  --chunk-size 1024
```

Result: `4,096` positions from `64` generated games, `4` files.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_selfplay_4096_t01_v1 data/teacher/alpha_recent120_fullgame_legalvalue_v2 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 data/teacher/alpha_recent120_fullgame_legalvalue_v2 \
  --out experiments/policyhead192-enginegames-policyhead-v1/checkpoints/iter_0001 \
  --epochs 3 \
  --batch-size 512 \
  --lr 5e-7 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.45 0.35 0.20 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_source_0_policy_acc \
  --device cuda
```

## Validation Read

Same validation command for parent and candidate:

```bash
uv run alpha-chess validate \
  --checkpoint <checkpoint> \
  --data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 data/teacher/stockfish_selfplay_4096_t01_v1 data/teacher/alpha_recent120_fullgame_legalvalue_v2 \
  --batch-size 512 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --device cuda
```

| Checkpoint | Overall top-1 | Broad holdout top-1 | Engine self-play top-1 | Recent120 v2 top-1 | Broad top-3/top-5 | Direct gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| parent `policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1` | `0.3231` | `0.3395` | `0.2480` | `0.3652` | `0.5421` / `0.6425` | N/A |
| `policyhead192-enginegames-policyhead-v1`, weights `0.45/0.35/0.20` | `0.3224` | `0.3390` | `0.2473` | `0.3645` | `0.5363` / `0.6385` | rejected before direct gate |
| `policyhead192-enginegames-lowweight-policyhead-v1`, weights `0.70/0.10/0.20` | `0.3226` | `0.3387` | `0.2458` | `0.3669` | `0.5366` / `0.6383` | rejected before direct gate |

Both candidates had lower policy losses than the parent, but the ranking metric
that matters for direct move choice got worse on the broad holdout. The
low-weight run suggests that simply reducing the new source weight is not
enough. The next useful variant should either make the engine-game source
stronger and disjoint, or train from it at larger scale with a selector that
requires broad holdout top-k to stay flat.
