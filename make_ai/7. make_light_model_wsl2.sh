#!/bin/bash
# WSL2에서 TensorFlow GPU를 사용해 Light 모델(7. make_light_model.py) 학습

cd /mnt/c/Users/slhg1/OneDrive/Desktop/SMARTCS_Pole

# 가상환경 활성화 (WSL2 TensorFlow GPU 설정 후 사용)
if [ -d "venv_wsl2" ]; then
  source venv_wsl2/bin/activate
else
  echo "venv_wsl2 없음. WSL2_TensorFlow_GPU_설정_가이드.md 따라 가상환경과 tensorflow[and-cuda] 설치 후 실행하세요."
  exit 1
fi

echo "GPU 확인 중..."
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

echo "Light 모델 학습 시작..."
python3 "make_ai/7. make_light_model.py" --local
