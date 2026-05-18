#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-l4}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-4}"
DATA_DIR="${DATA_DIR:-data/expert/lichess_2013_01_10k}"
OUT_DIR="${OUT_DIR:-checkpoints/expert_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CHANNELS="${CHANNELS:-128}"
BLOCKS="${BLOCKS:-6}"
LR="${LR:-0.001}"
RUN_NAME="${RUN_NAME:-expert-train-$(date +%Y%m%d-%H%M%S)}"

gpu-dev submit \
  --gpu-type "$GPU_TYPE" \
  --gpus "$GPUS" \
  --hours "$HOURS" \
  --runtime . \
  --name "alphachess-$RUN_NAME" \
  -- bash -lc "
    set -euo pipefail
    trap 'rm -r .venv 2>/dev/null || true' EXIT
    uv sync
    uv run alpha-chess train \
      --data $DATA_DIR \
      --out $OUT_DIR \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS \
      --lr $LR
  "
