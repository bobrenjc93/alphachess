# Stockfish Holdout Probe

Timestamp: `2026-05-19T21:33:15-07:00`

I added `--skip-positions` to `stockfish-teacher` and generated a disjoint
8,192-position Stockfish MultiPV holdout from the same raw Lichess source used
for the broad65k teacher set:

```text
data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536
```

Generation command:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
uv run alpha-chess stockfish-teacher \
  --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst \
  --out data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.005 \
  --multipv 4 \
  --policy-temperature-cp 200 \
  --min-elo 1800 \
  --min-initial-seconds 180 \
  --position-stride 2 \
  --skip-positions 65536 \
  --max-positions 8192 \
  --chunk-size 1024
```

Summary:

```text
games_seen=51539
games_used=224
skip_positions=65536
skipped_positions=65536
positions=8192
files=8
```

I validated representative checkpoints with hard best-move labels from the
holdout MultiPV files:

```bash
uv run alpha-chess validate \
  --checkpoint CHECKPOINT \
  --data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --batch-size 256 \
  --value-weight 0.10 \
  --legal-policy-loss \
  --prefer-action-labels \
  --device cpu
```

| Checkpoint | Policy loss | Top-1 | Top-3 | Top-5 | Value loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `focus-qvalue-puzzlelines20-vw025-material015` | `2.5938` | `0.3197` | `0.5643` | `0.6727` | `0.1025` |
| `policyhead-broad-qvalue-vw000-material015` | `2.5533` | `0.3206` | `0.5610` | `0.6740` | `0.1025` |
| `policyhead-broad32k-hardlabel-lossrepair-v1` | `2.4957` | `0.3252` | `0.5706` | `0.6801` | `0.1025` |
| `policyhead-broad65k-hardlabel-selectbest-v1` | `2.4966` | `0.3252` | `0.5704` | `0.6797` | `0.1025` |
| `policyhead-broad65k-expertmix-v1` | `2.4682` | `0.3260` | `0.5691` | `0.6803` | `0.1025` |
| `fullnet192-broad65k-expertmix-scratch-v1` | `3.2326` | `0.3381` | `0.5382` | `0.6395` | `0.1140` |
| `policyhead192-broad65k-puzzlemix-v1` | `2.8254` | `0.3429` | `0.5439` | `0.6443` | `0.1140` |
| `policyhead192-fullnetloss-cpu-overfit-v1` | `3.2747` | `0.3375` | `0.5380` | `0.6390` | `0.1140` |

Read:

- The larger `192x8` puzzle-mix checkpoint has the best hard-label top-1 on
  unseen broad Stockfish positions so far (`0.3429`), but its top-3/top-5 are
  materially worse than the `128x6` policy-head family.
- The `128x6` broad65k expert mix has the best policy loss and top-5 of this
  slice, but top-1 is still only `0.3260`.
- The CPU fullnet loss-overfit smoke regressed both direct play and this holdout
  versus the fullnet192 scratch parent.
- This is a better generalization diagnostic than validating on the broad65k
  training set, but it is still only supervised teacher agreement. The direct
  Stockfish promotion gate remains the required strength signal.
