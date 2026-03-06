#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -d "venv_gpu" ]; then
  # shellcheck disable=SC1091
  source venv_gpu/bin/activate
elif [ -d "venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source venv_wsl2/bin/activate
else
  echo "오류: venv_gpu 또는 venv_wsl2를 찾을 수 없습니다."
  exit 1
fi

echo "[정보] GPU 환경 확인"
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo "[정보] MLP 모델 학습 시작"
python3 "main/4. make_mlp_model/2. mlp_model.py" --local "$@"
echo "[정보] MLP 모델 학습 완료"
