#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-8}"
ITERATIONS="${ITERATIONS:-4}"
GAMES="${GAMES:-64}"
SIMULATIONS="${SIMULATIONS:-64}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CHANNELS="${CHANNELS:-128}"
BLOCKS="${BLOCKS:-6}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

gpu-dev submit \
  --gpu-type "$GPU_TYPE" \
  --gpus "$GPUS" \
  --hours "$HOURS" \
  --runtime . \
  --name "alphachess-iter-$RUN_NAME" \
  -- bash -lc "
    set -euo pipefail
    uv sync
    uv run alpha-chess iterate \
      --run-dir experiments/$RUN_NAME \
      --iterations $ITERATIONS \
      --games $GAMES \
      --simulations $SIMULATIONS \
      --eval-simulations $SIMULATIONS \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS
  "
