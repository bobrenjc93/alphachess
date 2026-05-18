#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-4}"
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
  --name "alphachess-$RUN_NAME" \
  -- bash -lc "
    set -euo pipefail
    uv sync
    uv run alpha-chess self-play \
      --games $GAMES \
      --simulations $SIMULATIONS \
      --out data/selfplay/$RUN_NAME
    uv run alpha-chess train \
      --data data/selfplay/$RUN_NAME \
      --out checkpoints/$RUN_NAME \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS
    uv run alpha-chess eval \
      --checkpoint checkpoints/$RUN_NAME/latest.pt \
      --games 4 \
      --simulations $SIMULATIONS
  "
