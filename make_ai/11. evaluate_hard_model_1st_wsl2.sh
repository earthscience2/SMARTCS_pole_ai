#!/bin/bash
# Evaluate hard 1st-stage bbox model (single run)

set -euo pipefail
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

cd /mnt/c/Users/slhg1/OneDrive/Desktop/SMARTCS_Pole

if [ -d "venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source venv_wsl2/bin/activate
else
  echo "venv_wsl2 not found. Run setup_wsl2_venv.sh first."
  exit 1
fi

echo "GPU check..."
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

TARGET_MEAN_IOU="${HARD1_TARGET_MEAN_IOU:-0.45}"
TARGET_IOU_05="${HARD1_TARGET_IOU_05:-0.55}"
TARGET_IOU_07="${HARD1_TARGET_IOU_07:-0.30}"
PASS_MODE="${HARD1_PASS_MODE:-average}"

echo "Evaluate hard 1st-stage bbox model..."
python3 "make_ai/11. evaluate_hard_model_1st.py" \
  --target-mean-best-iou "${TARGET_MEAN_IOU}" \
  --target-ratio-iou-0-5 "${TARGET_IOU_05}" \
  --target-ratio-iou-0-7 "${TARGET_IOU_07}" \
  --target-pass-mode "${PASS_MODE}" \
  "$@"
