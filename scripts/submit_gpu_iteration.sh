#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-8}"
ITERATIONS="${ITERATIONS:-4}"
CHECKPOINT="${CHECKPOINT:-}"
GAMES="${GAMES:-64}"
SIMULATIONS="${SIMULATIONS:-64}"
MAX_PLIES="${MAX_PLIES:-512}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CHANNELS="${CHANNELS:-128}"
BLOCKS="${BLOCKS:-6}"
LR="${LR:-0.001}"
LEGAL_POLICY_LOSS="${LEGAL_POLICY_LOSS:-0}"
REPLAY_DATA="${REPLAY_DATA:-}"
SELF_PLAY_WEIGHT="${SELF_PLAY_WEIGHT:-1.0}"
REPLAY_WEIGHTS="${REPLAY_WEIGHTS:-}"
EVAL_GAMES="${EVAL_GAMES:-8}"
EVAL_SIMULATIONS="${EVAL_SIMULATIONS:-$SIMULATIONS}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi
LEGAL_POLICY_LOSS_ARG=""
if [[ "$LEGAL_POLICY_LOSS" == "1" || "$LEGAL_POLICY_LOSS" == "true" ]]; then
  LEGAL_POLICY_LOSS_ARG="--legal-policy-loss"
fi
REPLAY_DATA_ARG=""
if [[ -n "$REPLAY_DATA" ]]; then
  REPLAY_DATA_ARG="--replay-data $REPLAY_DATA"
fi
REPLAY_WEIGHTS_ARG=""
if [[ -n "$REPLAY_WEIGHTS" ]]; then
  REPLAY_WEIGHTS_ARG="--replay-weights $REPLAY_WEIGHTS"
fi

gpu-dev submit \
  --gpu-type "$GPU_TYPE" \
  --gpus "$GPUS" \
  --hours "$HOURS" \
  --runtime . \
  --name "alphachess-iter-$RUN_NAME" \
  -- bash -lc "
    set -euo pipefail
    trap 'rm -r .venv 2>/dev/null || true' EXIT
    uv sync
    uv run alpha-chess iterate \
      --run-dir experiments/$RUN_NAME \
      --iterations $ITERATIONS \
      $CHECKPOINT_ARG \
      --games $GAMES \
      --simulations $SIMULATIONS \
      --max-plies $MAX_PLIES \
      --eval-games $EVAL_GAMES \
      --eval-simulations $EVAL_SIMULATIONS \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS \
      --lr $LR \
      --self-play-weight $SELF_PLAY_WEIGHT \
      $REPLAY_DATA_ARG \
      $REPLAY_WEIGHTS_ARG \
      $LEGAL_POLICY_LOSS_ARG
  "
