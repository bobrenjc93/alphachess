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
PROMOTION_SCORE="${PROMOTION_SCORE:-0.50}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

WORKSPACE_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/alphachess-iter-runtime.XXXXXX")"
trap 'rm -rf "$RUNTIME_DIR"' EXIT

copy_runtime_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return
  fi
  if [[ "$path" = /* ]]; then
    echo "Runtime paths must be relative to the repository: $path" >&2
    exit 1
  fi
  if [[ ! -e "$WORKSPACE_ROOT/$path" ]]; then
    echo "Missing runtime path: $path" >&2
    exit 1
  fi
  mkdir -p "$RUNTIME_DIR/$(dirname "$path")"
  if [[ -d "$WORKSPACE_ROOT/$path" ]]; then
    mkdir -p "$RUNTIME_DIR/$path"
    rsync -a --delete "$WORKSPACE_ROOT/$path"/ "$RUNTIME_DIR/$path"/
  else
    rsync -a "$WORKSPACE_ROOT/$path" "$RUNTIME_DIR/$path"
  fi
}

rsync -a --delete \
  --exclude .git/ \
  --exclude .venv/ \
  --exclude __pycache__/ \
  --exclude .pytest_cache/ \
  --exclude .ruff_cache/ \
  --exclude data/ \
  --exclude checkpoints/ \
  --exclude experiments/ \
  "$WORKSPACE_ROOT"/ "$RUNTIME_DIR"/

if [[ -e "$WORKSPACE_ROOT/experiments/$RUN_NAME" ]]; then
  copy_runtime_path "experiments/$RUN_NAME"
fi
copy_runtime_path "$CHECKPOINT"
for path in $REPLAY_DATA; do
  copy_runtime_path "$path"
done

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

submit_status=0
gpu-dev submit \
  --gpu-type "$GPU_TYPE" \
  --gpus "$GPUS" \
  --hours "$HOURS" \
  --runtime "$RUNTIME_DIR" \
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
      --promotion-score $PROMOTION_SCORE \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS \
      --lr $LR \
      --self-play-weight $SELF_PLAY_WEIGHT \
      $REPLAY_DATA_ARG \
      $REPLAY_WEIGHTS_ARG \
      $LEGAL_POLICY_LOSS_ARG
  " || submit_status=$?

if [[ -d "$RUNTIME_DIR/experiments/$RUN_NAME" ]]; then
  mkdir -p "$WORKSPACE_ROOT/experiments"
  rsync -a --delete "$RUNTIME_DIR/experiments/$RUN_NAME" "$WORKSPACE_ROOT/experiments/"
fi
exit "$submit_status"
