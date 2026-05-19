#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-8}"
ITERATIONS="${ITERATIONS:-4}"
CHECKPOINT="${CHECKPOINT:-}"
GAMES="${GAMES:-64}"
SELF_PLAY_WORKERS="${SELF_PLAY_WORKERS:-1}"
SIMULATIONS="${SIMULATIONS:-64}"
C_PUCT="${C_PUCT:-1.5}"
POLICY_PRIOR_TEMPERATURE="${POLICY_PRIOR_TEMPERATURE:-1.0}"
TREE_REUSE="${TREE_REUSE:-1}"
MAX_PLIES="${MAX_PLIES:-512}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CHANNELS="${CHANNELS:-128}"
BLOCKS="${BLOCKS:-6}"
LR="${LR:-0.001}"
VALUE_WEIGHT="${VALUE_WEIGHT:-1.0}"
BAD_ACTION_WEIGHT="${BAD_ACTION_WEIGHT:-0.0}"
BAD_ACTION_MARGIN="${BAD_ACTION_MARGIN:-1.0}"
LEGAL_POLICY_LOSS="${LEGAL_POLICY_LOSS:-0}"
COLOR_MIRROR_AUGMENTATION="${COLOR_MIRROR_AUGMENTATION:-0}"
PREFER_ACTION_LABELS="${PREFER_ACTION_LABELS:-0}"
POLICY_HEAD_ONLY="${POLICY_HEAD_ONLY:-0}"
VALUE_HEAD_ONLY="${VALUE_HEAD_ONLY:-0}"
MATERIAL_VALUE_WEIGHT="${MATERIAL_VALUE_WEIGHT:-0.0}"
MATERIAL_VALUE_SEARCH_PLIES="${MATERIAL_VALUE_SEARCH_PLIES:-0}"
ROOT_MATE_SEARCH_PLIES="${ROOT_MATE_SEARCH_PLIES:-3}"
ROOT_MATERIAL_SEARCH_PLIES="${ROOT_MATERIAL_SEARCH_PLIES:-0}"
ROOT_MATERIAL_MAX_LOSS_CP="${ROOT_MATERIAL_MAX_LOSS_CP:-250}"
REPLAY_DATA="${REPLAY_DATA:-}"
SELF_PLAY_WEIGHT="${SELF_PLAY_WEIGHT:-1.0}"
REPLAY_WEIGHTS="${REPLAY_WEIGHTS:-}"
SELF_PLAY_POLICY_WEIGHT="${SELF_PLAY_POLICY_WEIGHT:-1.0}"
REPLAY_POLICY_WEIGHTS="${REPLAY_POLICY_WEIGHTS:-}"
EVAL_GAMES="${EVAL_GAMES:-8}"
EVAL_SIMULATIONS="${EVAL_SIMULATIONS:-$SIMULATIONS}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
PROMOTION_SCORE="${PROMOTION_SCORE:-0.50}"
STOCKFISH_GATE_GAMES="${STOCKFISH_GATE_GAMES:-0}"
STOCKFISH_GATE_SIMULATIONS="${STOCKFISH_GATE_SIMULATIONS:-}"
STOCKFISH_GATE_MIN_SCORE="${STOCKFISH_GATE_MIN_SCORE:-0.50}"
STOCKFISH_GATE_ENGINE_PATH="${STOCKFISH_GATE_ENGINE_PATH:-stockfish}"
STOCKFISH_GATE_ENGINE_TIME="${STOCKFISH_GATE_ENGINE_TIME:-0.05}"
STOCKFISH_GATE_ENGINE_DEPTH="${STOCKFISH_GATE_ENGINE_DEPTH:-}"
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
COLOR_MIRROR_AUGMENTATION_ARG=""
if [[ "$COLOR_MIRROR_AUGMENTATION" == "1" || "$COLOR_MIRROR_AUGMENTATION" == "true" ]]; then
  COLOR_MIRROR_AUGMENTATION_ARG="--color-mirror-augmentation"
fi
PREFER_ACTION_LABELS_ARG=""
if [[ "$PREFER_ACTION_LABELS" == "1" || "$PREFER_ACTION_LABELS" == "true" ]]; then
  PREFER_ACTION_LABELS_ARG="--prefer-action-labels"
fi
POLICY_HEAD_ONLY_ARG=""
if [[ "$POLICY_HEAD_ONLY" == "1" || "$POLICY_HEAD_ONLY" == "true" ]]; then
  POLICY_HEAD_ONLY_ARG="--policy-head-only"
fi
VALUE_HEAD_ONLY_ARG=""
if [[ "$VALUE_HEAD_ONLY" == "1" || "$VALUE_HEAD_ONLY" == "true" ]]; then
  VALUE_HEAD_ONLY_ARG="--value-head-only"
fi
REPLAY_DATA_ARG=""
if [[ -n "$REPLAY_DATA" ]]; then
  REPLAY_DATA_ARG="--replay-data $REPLAY_DATA"
fi
REPLAY_WEIGHTS_ARG=""
if [[ -n "$REPLAY_WEIGHTS" ]]; then
  REPLAY_WEIGHTS_ARG="--replay-weights $REPLAY_WEIGHTS"
fi
REPLAY_POLICY_WEIGHTS_ARG=""
if [[ -n "$REPLAY_POLICY_WEIGHTS" ]]; then
  REPLAY_POLICY_WEIGHTS_ARG="--replay-policy-weights $REPLAY_POLICY_WEIGHTS"
fi
NO_TREE_REUSE_ARG=""
if [[ "$TREE_REUSE" == "0" || "$TREE_REUSE" == "false" ]]; then
  NO_TREE_REUSE_ARG="--no-tree-reuse"
fi
STOCKFISH_GATE_SIMULATIONS_ARG=""
if [[ -n "$STOCKFISH_GATE_SIMULATIONS" ]]; then
  STOCKFISH_GATE_SIMULATIONS_ARG="--stockfish-gate-simulations $STOCKFISH_GATE_SIMULATIONS"
fi
STOCKFISH_GATE_ENGINE_DEPTH_ARG=""
if [[ -n "$STOCKFISH_GATE_ENGINE_DEPTH" ]]; then
  STOCKFISH_GATE_ENGINE_DEPTH_ARG="--stockfish-gate-engine-depth $STOCKFISH_GATE_ENGINE_DEPTH"
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
      --self-play-workers $SELF_PLAY_WORKERS \
      --simulations $SIMULATIONS \
      --c-puct $C_PUCT \
      --policy-prior-temperature $POLICY_PRIOR_TEMPERATURE \
      $NO_TREE_REUSE_ARG \
      --max-plies $MAX_PLIES \
      --eval-games $EVAL_GAMES \
      --eval-simulations $EVAL_SIMULATIONS \
      --eval-workers $EVAL_WORKERS \
      --promotion-score $PROMOTION_SCORE \
      --stockfish-gate-games $STOCKFISH_GATE_GAMES \
      $STOCKFISH_GATE_SIMULATIONS_ARG \
      --stockfish-gate-min-score $STOCKFISH_GATE_MIN_SCORE \
      --stockfish-gate-engine-path $STOCKFISH_GATE_ENGINE_PATH \
      --stockfish-gate-engine-time $STOCKFISH_GATE_ENGINE_TIME \
      $STOCKFISH_GATE_ENGINE_DEPTH_ARG \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --channels $CHANNELS \
      --blocks $BLOCKS \
      --lr $LR \
      --value-weight $VALUE_WEIGHT \
      --bad-action-weight $BAD_ACTION_WEIGHT \
      --bad-action-margin $BAD_ACTION_MARGIN \
      --material-value-weight $MATERIAL_VALUE_WEIGHT \
      --material-value-search-plies $MATERIAL_VALUE_SEARCH_PLIES \
      --root-mate-search-plies $ROOT_MATE_SEARCH_PLIES \
      --root-material-search-plies $ROOT_MATERIAL_SEARCH_PLIES \
      --root-material-max-loss-cp $ROOT_MATERIAL_MAX_LOSS_CP \
      --self-play-weight $SELF_PLAY_WEIGHT \
      --self-play-policy-weight $SELF_PLAY_POLICY_WEIGHT \
      $REPLAY_DATA_ARG \
      $REPLAY_WEIGHTS_ARG \
      $REPLAY_POLICY_WEIGHTS_ARG \
      $LEGAL_POLICY_LOSS_ARG \
      $COLOR_MIRROR_AUGMENTATION_ARG \
      $PREFER_ACTION_LABELS_ARG \
      $POLICY_HEAD_ONLY_ARG \
      $VALUE_HEAD_ONLY_ARG
  " || submit_status=$?

if [[ -d "$RUNTIME_DIR/experiments/$RUN_NAME" ]]; then
  mkdir -p "$WORKSPACE_ROOT/experiments"
  rsync -a --delete "$RUNTIME_DIR/experiments/$RUN_NAME" "$WORKSPACE_ROOT/experiments/"
fi
exit "$submit_status"
