#!/bin/bash
# Hard 1st-stage loop: train -> evaluate -> evaluation-based retrain

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

MAX_ROUNDS="${HARD1_MAX_ROUNDS:-3}"
EPOCHS="${HARD1_EPOCHS:-300}"
BATCH_SIZE="${HARD1_BATCH_SIZE:-32}"
EXP_SCHEDULE_STR="${HARD1_EXP_SCHEDULE:-0 1 6}"
TARGET_MEAN_IOU="${HARD1_TARGET_MEAN_IOU:-0.45}"
TARGET_IOU_05="${HARD1_TARGET_IOU_05:-0.55}"
TARGET_IOU_07="${HARD1_TARGET_IOU_07:-0.30}"
PASS_MODE="${HARD1_PASS_MODE:-average}"

read -r -a EXP_SCHEDULE <<< "$EXP_SCHEDULE_STR"
if [ "${#EXP_SCHEDULE[@]}" -eq 0 ]; then
  echo "HARD1_EXP_SCHEDULE is empty."
  exit 1
fi

LATEST_RUN=""
PASSED=0

for ((round=1; round<=MAX_ROUNDS; round++)); do
  idx=$(( (round - 1) % ${#EXP_SCHEDULE[@]} ))
  exp_id="${EXP_SCHEDULE[$idx]}"
  run_tag="loop1_r${round}"

  echo "[Round ${round}/${MAX_ROUNDS}] Train hard 1st-stage (exp=${exp_id}, epochs=${EPOCHS}, batch=${BATCH_SIZE})"
  python3 "make_ai/10. make_hard_model_1st.py" --local \
    --exp "${exp_id}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --run-tag "${run_tag}"

  LATEST_RUN="$(python3 - <<'PY'
from pathlib import Path
base = Path('make_ai/10. hard_models_1st')
runs = [d for d in base.iterdir() if d.is_dir() and (d / 'checkpoints' / 'best_x.keras').exists()]
print(str(max(runs, key=lambda d: d.name).resolve()) if runs else '')
PY
)"
  if [ -z "$LATEST_RUN" ]; then
    echo "No trained run found under make_ai/10. hard_models_1st"
    exit 1
  fi

  echo "[Round ${round}/${MAX_ROUNDS}] Evaluate run: ${LATEST_RUN}"
  python3 "make_ai/11. evaluate_hard_model_1st.py" \
    --run_dir "${LATEST_RUN}" \
    --target-mean-best-iou "${TARGET_MEAN_IOU}" \
    --target-ratio-iou-0-5 "${TARGET_IOU_05}" \
    --target-ratio-iou-0-7 "${TARGET_IOU_07}" \
    --target-pass-mode "${PASS_MODE}"

  FEEDBACK_PATH="$(python3 - <<'PY'
from pathlib import Path
base = Path('make_ai/11. evaluate_hard_model_1st')
feedbacks = sorted(base.glob('*/training_feedback.json'))
print(str(feedbacks[-1].resolve()) if feedbacks else '')
PY
)"
  if [ -z "$FEEDBACK_PATH" ]; then
    echo "training_feedback.json not found after evaluation."
    exit 1
  fi

  PASSED="$(python3 - "$FEEDBACK_PATH" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
print('1' if data.get('pass') else '0')
PY
)"

  if [ "$PASSED" = "1" ]; then
    echo "[Round ${round}] PASS -> stop retraining loop."
    python3 - "$LATEST_RUN" <<'PY'
import shutil
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path('make_ai/hard_model_1st_best')
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'Updated best alias: {dst}')
PY
    break
  fi

  echo "[Round ${round}] FAIL -> continue retraining."
done

if [ "$PASSED" != "1" ]; then
  echo "Hard 1st-stage loop finished without meeting targets."
  exit 2
fi

echo "Hard 1st-stage loop complete."
