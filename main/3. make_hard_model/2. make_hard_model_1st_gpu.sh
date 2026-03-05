#!/bin/bash
set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -d "venv_gpu" ]; then
  # shellcheck disable=SC1091
  source venv_gpu/bin/activate
elif [ -d "venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source venv_wsl2/bin/activate
elif [ -d "$SCRIPT_DIR/../../venv_gpu" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../venv_gpu/bin/activate"
elif [ -d "$SCRIPT_DIR/../../venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../../venv_wsl2/bin/activate"
else
  echo "venv_gpu 또는 venv_wsl2가 없습니다. WSL에서 가상환경 생성 후 tensorflow 설치가 필요합니다."
  exit 1
fi

echo "GPU check..."
if ! python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))" 2>/dev/null; then
  echo "오류: 현재 가상환경에 tensorflow가 설치되어 있지 않습니다."
  echo "  source venv_gpu/bin/activate && pip install tensorflow   # 또는 'tensorflow[and-cuda]'"
  exit 1
fi

EPOCHS="${HARD1_EPOCHS:-300}"
BATCH_SIZE="${HARD1_BATCH_SIZE:-32}"
TARGET_MEAN_IOU="${HARD1_TARGET_MEAN_IOU:-0.45}"
TARGET_IOU_05="${HARD1_TARGET_IOU_05:-0.55}"
TARGET_IOU_07="${HARD1_TARGET_IOU_07:-0.30}"
PASS_MODE="${HARD1_PASS_MODE:-average}"

python3 "main/3. make_hard_model/2. make_hard_model_1st.py" --local \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --target-mean-best-iou "$TARGET_MEAN_IOU" \
  --target-ratio-iou-0-5 "$TARGET_IOU_05" \
  --target-ratio-iou-0-7 "$TARGET_IOU_07" \
  --target-pass-mode "$PASS_MODE"
