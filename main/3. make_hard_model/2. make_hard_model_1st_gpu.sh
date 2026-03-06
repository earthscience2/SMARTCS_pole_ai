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
  echo "오류: venv_gpu 또는 venv_wsl2를 찾을 수 없습니다. WSL 환경에서 가상환경 생성 후 TensorFlow를 설치하세요."
  exit 1
fi

echo "[정보] GPU 환경 확인"
if ! python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))" 2>/dev/null; then
  echo "오류: WSL 가상환경에 tensorflow가 설치되어 있지 않습니다."
  echo "  source venv_gpu/bin/activate && pip install tensorflow   # GPU 환경이면 'tensorflow[and-cuda]' 검토"
  exit 1
fi

EPOCHS="${HARD1_EPOCHS:-300}"
BATCH_SIZE="${HARD1_BATCH_SIZE:-32}"

python3 "main/3. make_hard_model/2. make_hard_model_1st.py" --local \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"
