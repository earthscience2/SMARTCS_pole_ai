#!/bin/bash
# Light model loop: train -> evaluate -> evaluation-based retrain

set -euo pipefail

# 프로젝트 루트로 이동 (경로는 환경에 맞게 수정)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -d "venv_gpu" ]; then
  # shellcheck disable=SC1091
  source venv_gpu/bin/activate
elif [ -d "venv_wsl2" ]; then
  source venv_wsl2/bin/activate
else
  echo "venv_gpu 또는 venv_wsl2 가 없습니다. WSL에서 가상환경 생성 후 tensorflow 설치가 필요합니다."
  exit 1
fi

echo "GPU check..."
if ! python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))" 2>/dev/null; then
  echo ""
  echo "오류: tensorflow가 현재 가상환경에 설치되어 있지 않습니다."
  echo "WSL 터미널에서 프로젝트 루트로 이동한 뒤 아래를 실행하세요:"
  echo "  source venv_gpu/bin/activate   # 또는 venv_wsl2"
  echo "  pip install tensorflow         # GPU 사용 시: pip install 'tensorflow[and-cuda]'"
  exit 1
fi

MAX_ROUNDS="${LIGHT_MAX_ROUNDS:-1}"
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
  echo "[Round ${round}/${MAX_ROUNDS}] Train + Evaluate (epochs=${epochs}, batch=${BATCH_SIZE}, lr=${lr}, alpha=${focal_alpha}, break_w=${break_weight_scale})"
  python3 "main/2. make_light_model/2. make_light_model.py" --local \
    --epochs "${epochs}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${lr}" \
    --focal-alpha "${focal_alpha}" \
    --break-class-weight-scale "${break_weight_scale}" \
    --target-precision "${TARGET_PRECISION}" \
    --target-recall "${TARGET_RECALL}" \
    --target-f1 "${TARGET_F1}" \
    --target-pass-mode "${PASS_MODE}"

  LATEST_RUN_NAME="$(python3 - <<'PY'
from pathlib import Path
base = Path('main/2. make_light_model/2. light_models')
runs = [d for d in base.iterdir() if d.is_dir() and (d / 'checkpoints' / 'best.keras').exists()]
print(max(runs, key=lambda d: d.name).name if runs else '')
PY
)"
  if [ -z "$LATEST_RUN_NAME" ]; then
    echo "No trained run found under main/2. make_light_model/2. light_models"
    exit 1
  fi

  # 평가 결과는 2. light_models/<run>/evaluation/ 에 저장됨
  FEEDBACK_PATH="main/2. make_light_model/2. light_models/${LATEST_RUN_NAME}/evaluation/training_feedback.json"
  if [ ! -f "$FEEDBACK_PATH" ]; then
    echo "training_feedback.json not found: $FEEDBACK_PATH"
    exit 1
  fi
  FEEDBACK_PATH="$(cd "$(dirname "$FEEDBACK_PATH")" && pwd)/$(basename "$FEEDBACK_PATH")"
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
src = Path('main/2. make_light_model/2. light_models') / sys.argv[1]
dst = Path('main/2. make_light_model/best_light_model')
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst / sys.argv[1])
print(f'Updated best alias: {dst / sys.argv[1]}')
PY
    break
  fi

  if [ "${round}" -lt "${MAX_ROUNDS}" ]; then
    echo "[Round ${round}] FAIL -> continue retraining."
  else
    echo "[Round ${round}] FAIL -> stop (no retrain by default; set LIGHT_MAX_ROUNDS>1 to enable loop)."
  fi
done

if [ "$PASSED" != "1" ]; then
  echo "Light model loop finished without meeting targets."
  exit 2
fi

echo "Light model loop complete."
