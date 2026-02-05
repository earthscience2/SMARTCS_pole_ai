#!/bin/bash
# WSL2에서 TensorFlow GPU를 사용해 Hard 2nd-stage(conf head) 모델 평가(13. evaluate_hard_model_2nd.py)

cd /mnt/c/Users/slhg1/OneDrive/Desktop/SMARTCS_Pole

# 가상환경 활성화 (WSL2 TensorFlow GPU 설정 후 사용)
if [ -d "venv_wsl2" ]; then
  source venv_wsl2/bin/activate
else
  echo "venv_wsl2 없음. setup_wsl2_venv.sh 실행 후 tensorflow[and-cuda] 설치 필요."
  exit 1
fi

echo "GPU 확인 중..."
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo "Hard 2nd-stage(conf head) 모델 평가 시작..."
python3 "make_ai/13. evaluate_hard_model_2nd.py" --local "$@"

