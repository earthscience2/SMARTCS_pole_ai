#!/bin/bash
# Light model loop: train -> evaluate -> evaluation-based retrain

set -euo pipefail

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

MAX_ROUNDS="${LIGHT_MAX_ROUNDS:-3}"
EPOCHS_BASE="${LIGHT_EPOCHS:-100}"
EPOCHS_STEP="${LIGHT_EPOCHS_STEP:-20}"
BATCH_SIZE="${LIGHT_BATCH_SIZE:-32}"
LR_BASE="${LIGHT_LR:-0.001}"
LR_DECAY="${LIGHT_LR_DECAY:-0.9}"
FOCAL_ALPHA_SCHEDULE_STR="${LIGHT_FOCAL_ALPHA_SCHEDULE:-0.93 0.95 0.97}"
BREAK_WEIGHT_SCALE_SCHEDULE_STR="${LIGHT_BREAK_WEIGHT_SCALE_SCHEDULE:-1.35 1.45 1.55}"
TARGET_PRECISION="${LIGHT_TARGET_PRECISION:-0.80}"
TARGET_RECALL="${LIGHT_TARGET_RECALL:-0.90}"
TARGET_F1="${LIGHT_TARGET_F1:-0.84}"
PASS_MODE="${LIGHT_PASS_MODE:-all_metrics}"

read -r -a FOCAL_ALPHA_SCHEDULE <<< "$FOCAL_ALPHA_SCHEDULE_STR"
read -r -a BREAK_WEIGHT_SCALE_SCHEDULE <<< "$BREAK_WEIGHT_SCALE_SCHEDULE_STR"
if [ "${#FOCAL_ALPHA_SCHEDULE[@]}" -eq 0 ] || [ "${#BREAK_WEIGHT_SCALE_SCHEDULE[@]}" -eq 0 ]; then
  echo "Schedule variables are empty."
  exit 1
fi

LATEST_RUN_NAME=""
PASSED=0

for ((round=1; round<=MAX_ROUNDS; round++)); do
  idx_alpha=$(( (round - 1) % ${#FOCAL_ALPHA_SCHEDULE[@]} ))
  idx_weight=$(( (round - 1) % ${#BREAK_WEIGHT_SCALE_SCHEDULE[@]} ))
  focal_alpha="${FOCAL_ALPHA_SCHEDULE[$idx_alpha]}"
  break_weight_scale="${BREAK_WEIGHT_SCALE_SCHEDULE[$idx_weight]}"
  epochs=$((EPOCHS_BASE + (round - 1) * EPOCHS_STEP))
  lr="$(python3 - <<PY
base = float("${LR_BASE}")
decay = float("${LR_DECAY}")
round_idx = ${round} - 1
print(f"{base * (decay ** round_idx):.8f}")
PY
)"
  run_tag="loopL_r${round}"

  echo "[Round ${round}/${MAX_ROUNDS}] Train light model (epochs=${epochs}, batch=${BATCH_SIZE}, lr=${lr}, alpha=${focal_alpha}, break_w=${break_weight_scale})"
  python3 "make_ai/7. make_light_model.py" --local \
    --epochs "${epochs}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${lr}" \
    --focal-alpha "${focal_alpha}" \
    --break-class-weight-scale "${break_weight_scale}" \
    --run-tag "${run_tag}"

  LATEST_RUN_NAME="$(python3 - <<'PY'
from pathlib import Path
base = Path('make_ai/7. light_models')
runs = [d for d in base.iterdir() if d.is_dir() and (d / 'checkpoints' / 'best.keras').exists()]
print(max(runs, key=lambda d: d.name).name if runs else '')
PY
)"
  if [ -z "$LATEST_RUN_NAME" ]; then
    echo "No trained run found under make_ai/7. light_models"
    exit 1
  fi

  echo "[Round ${round}/${MAX_ROUNDS}] Evaluate run: ${LATEST_RUN_NAME}"
  python3 "make_ai/8. evaluate_light_model.py" \
    --run "${LATEST_RUN_NAME}" \
    --target-precision "${TARGET_PRECISION}" \
    --target-recall "${TARGET_RECALL}" \
    --target-f1 "${TARGET_F1}" \
    --target-pass-mode "${PASS_MODE}"

  FEEDBACK_PATH="$(python3 - "$LATEST_RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path
run_name = sys.argv[1]
base = Path('make_ai/8. evaluate_light_model')
candidates = []
for p in base.glob('*/training_feedback.json'):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('model_run') == run_name:
            candidates.append((p.stat().st_mtime, p))
    except Exception:
        pass
candidates.sort(key=lambda x: x[0])
print(str(candidates[-1][1].resolve()) if candidates else '')
PY
)"
  if [ -z "$FEEDBACK_PATH" ]; then
    echo "training_feedback.json not found after light evaluation."
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
    python3 - "$LATEST_RUN_NAME" <<'PY'
import shutil
import sys
from pathlib import Path
src = Path('make_ai/7. light_models') / sys.argv[1]
dst = Path('make_ai/light_model_best')
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
  echo "Light model loop finished without meeting targets."
  exit 2
fi

echo "Light model loop complete."
