#!/bin/bash
set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

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
  echo "오류: venv_gpu 또는 venv_wsl2를 찾을 수 없습니다. WSL 환경에서 가상환경 생성 후 TensorFlow를 설치하세요."
  exit 1
fi

echo "[정보] GPU 환경 확인"
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

FIRST_STAGE_RUN_DIR="${HARD2_FIRST_STAGE_RUN_DIR:-$SCRIPT_DIR/best_hard_model_1st}"
FIRST_STAGE_MODEL="${HARD2_FIRST_STAGE_MODEL:-}"
EPOCHS="${HARD2_EPOCHS:-220}"
BATCH_SIZE="${HARD2_BATCH_SIZE:-32}"
LR="${HARD2_LR:-0.001}"

PY_ARGS=(
  "--first-stage-run-dir" "$FIRST_STAGE_RUN_DIR"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--learning-rate" "$LR"
)

if [ -n "$FIRST_STAGE_MODEL" ]; then
  PY_ARGS+=("--first-stage-model" "$FIRST_STAGE_MODEL")
fi

python3 "main/3. make_hard_model/3. make_hard_model_2nd.py" --local "${PY_ARGS[@]}"
