#!/bin/bash
# Hard 2nd-stage loop: train conf-head -> evaluate -> evaluation-based retrain

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

MAX_ROUNDS="${HARD2_MAX_ROUNDS:-3}"
EPOCHS_BASE="${HARD2_EPOCHS:-220}"
EPOCHS_STEP="${HARD2_EPOCHS_STEP:-40}"
BATCH_SIZE="${HARD2_BATCH_SIZE:-32}"
LR_BASE="${HARD2_LR:-0.001}"
LR_DECAY="${HARD2_LR_DECAY:-0.8}"
FIRST_STAGE_RUN_DIR="${HARD2_FIRST_STAGE_RUN_DIR:-make_ai/hard_model_1st_best}"
TARGET_F1="${HARD2_TARGET_OVERALL_BEST_F1:-0.70}"
TARGET_AUC="${HARD2_TARGET_OVERALL_AUC:-0.70}"
TARGET_SEP="${HARD2_TARGET_OVERALL_SEPARATION:-0.20}"
PASS_MODE="${HARD2_PASS_MODE:-all_metrics}"

LATEST_RUN=""
PASSED=0

for ((round=1; round<=MAX_ROUNDS; round++)); do
  epochs=$((EPOCHS_BASE + (round - 1) * EPOCHS_STEP))
  lr="$(python3 - <<PY
base = float("${LR_BASE}")
decay = float("${LR_DECAY}")
round_idx = ${round} - 1
print(f"{base * (decay ** round_idx):.8f}")
PY
)"
  run_tag="loop2_r${round}"

  echo "[Round ${round}/${MAX_ROUNDS}] Train hard 2nd-stage (epochs=${epochs}, batch=${BATCH_SIZE}, lr=${lr})"
  python3 "make_ai/12. make_hard_model_2nd.py" --local \
    --first-stage-run-dir "${FIRST_STAGE_RUN_DIR}" \
    --epochs "${epochs}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${lr}" \
    --run-tag "${run_tag}"

  LATEST_RUN="$(python3 - <<'PY'
from pathlib import Path
base = Path('make_ai/12. hard_models_2nd')
runs = [d for d in base.iterdir() if d.is_dir() and (d / 'checkpoints' / 'conf_x.keras').exists()]
print(str(max(runs, key=lambda d: d.name).resolve()) if runs else '')
PY
)"
  if [ -z "$LATEST_RUN" ]; then
    echo "No trained run found under make_ai/12. hard_models_2nd"
    exit 1
  fi

  echo "[Round ${round}/${MAX_ROUNDS}] Evaluate run: ${LATEST_RUN}"
  python3 "make_ai/13. evaluate_hard_model_2nd.py" --local \
    --run-dir "${LATEST_RUN}" \
    --test-only \
    --target-overall-best-f1 "${TARGET_F1}" \
    --target-overall-auc "${TARGET_AUC}" \
    --target-overall-separation "${TARGET_SEP}" \
    --target-pass-mode "${PASS_MODE}"

  FEEDBACK_PATH="$(python3 - "$LATEST_RUN" <<'PY'
import json
import sys
from pathlib import Path
run_name = Path(sys.argv[1]).name
base = Path('make_ai/13. evaluate_hard_model_2nd')
candidates = []
for p in base.rglob('training_feedback.json'):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('second_stage_run') == run_name:
            candidates.append((p.stat().st_mtime, p))
    except Exception:
        pass
candidates.sort(key=lambda x: x[0])
print(str(candidates[-1][1].resolve()) if candidates else '')
PY
)"
  if [ -z "$FEEDBACK_PATH" ]; then
    echo "training_feedback.json not found after 2nd-stage evaluation."
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
dst = Path('make_ai/hard_model_2nd_best')
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
  echo "Hard 2nd-stage loop finished without meeting targets."
  exit 2
fi

echo "Hard 2nd-stage loop complete."
