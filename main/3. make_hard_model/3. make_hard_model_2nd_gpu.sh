#!/bin/bash
set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv_gpu" ]; then
  # shellcheck disable=SC1091
  source venv_gpu/bin/activate
elif [ -d "$SCRIPT_DIR/../../venv_gpu" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../venv_gpu/bin/activate"
elif [ -d "venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source venv_wsl2/bin/activate
elif [ -d "$SCRIPT_DIR/../../venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../venv_wsl2/bin/activate"
else
  echo "GPU 가상환경을 찾을 수 없습니다. venv_gpu 또는 venv_wsl2를 먼저 설정하세요."
  exit 1
fi

echo "GPU check..."
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

FIRST_STAGE_RUN_DIR="${HARD2_FIRST_STAGE_RUN_DIR:-$SCRIPT_DIR/best_hard_model_1st}"
FIRST_STAGE_MODEL="${HARD2_FIRST_STAGE_MODEL:-}"
EPOCHS="${HARD2_EPOCHS:-220}"
BATCH_SIZE="${HARD2_BATCH_SIZE:-32}"
LR="${HARD2_LR:-0.001}"
TARGET_F1="${HARD2_TARGET_OVERALL_BEST_F1:-0.70}"
TARGET_AUC="${HARD2_TARGET_OVERALL_AUC:-0.70}"
TARGET_SEP="${HARD2_TARGET_OVERALL_SEPARATION:-0.20}"
PASS_MODE="${HARD2_PASS_MODE:-all_metrics}"

# 명령어 구성
CMD_ARGS="--local"
CMD_ARGS="$CMD_ARGS --first-stage-run-dir \"$FIRST_STAGE_RUN_DIR\""
if [ -n "$FIRST_STAGE_MODEL" ]; then
  CMD_ARGS="$CMD_ARGS --first-stage-model \"$FIRST_STAGE_MODEL\""
fi
CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
CMD_ARGS="$CMD_ARGS --batch-size $BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --learning-rate $LR"
CMD_ARGS="$CMD_ARGS --target-overall-best-f1 $TARGET_F1"
CMD_ARGS="$CMD_ARGS --target-overall-auc $TARGET_AUC"
CMD_ARGS="$CMD_ARGS --target-overall-separation $TARGET_SEP"
CMD_ARGS="$CMD_ARGS --target-pass-mode $PASS_MODE"

eval "python3 \"$SCRIPT_DIR/3. make_hard_model_2nd.py\" $CMD_ARGS"

