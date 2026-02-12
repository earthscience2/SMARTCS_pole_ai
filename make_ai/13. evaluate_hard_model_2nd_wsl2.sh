#!/bin/bash
# Evaluate hard 2nd-stage(conf head) model (single run)

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

TARGET_F1="${HARD2_TARGET_OVERALL_BEST_F1:-0.70}"
TARGET_AUC="${HARD2_TARGET_OVERALL_AUC:-0.70}"
TARGET_SEP="${HARD2_TARGET_OVERALL_SEPARATION:-0.20}"
PASS_MODE="${HARD2_PASS_MODE:-all_metrics}"

echo "Evaluate hard 2nd-stage conf model..."
python3 "make_ai/13. evaluate_hard_model_2nd.py" --local \
  --target-overall-best-f1 "${TARGET_F1}" \
  --target-overall-auc "${TARGET_AUC}" \
  --target-overall-separation "${TARGET_SEP}" \
  --target-pass-mode "${PASS_MODE}" \
  "$@"
