#!/bin/bash
# MLP Model Training GPU Script

set -e

# 프로젝트 루트 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== MLP Model Training (GPU) ==="
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"

# Python 경로 찾기 (WSL2 환경)
PYTHON_CMD=""

# 1. 가상환경에서 Python 찾기
if [ -f "venv/bin/activate" ]; then
    echo "Activating Linux virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating Windows virtual environment in WSL..."
    # Windows 가상환경의 Python을 WSL에서 직접 실행
    PYTHON_CMD="venv/Scripts/python.exe"
else
    echo "Warning: Virtual environment not found, using system python"
    PYTHON_CMD="python3"
fi

# Python 경로 확인
echo "Python command: $PYTHON_CMD"
if command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Python path: $(which $PYTHON_CMD 2>/dev/null || echo 'Direct path')"
    echo "Python version: $($PYTHON_CMD --version 2>/dev/null || echo 'Version check failed')"
else
    echo "Python not found, trying alternatives..."
    # 대안 경로들 시도
    if [ -f "venv/Scripts/python.exe" ]; then
        PYTHON_CMD="./venv/Scripts/python.exe"
        echo "Using Windows Python: $PYTHON_CMD"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "Using system python3"
    else
        echo "Error: No Python found!"
        exit 1
    fi
fi

# CUDA 확인
echo "CUDA devices:"
$PYTHON_CMD -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))" 2>/dev/null || echo "TensorFlow check failed"

# MLP 모델 훈련 실행
echo "Starting MLP model training..."
$PYTHON_CMD "main/4. make_mlp_model/2. mlp_model.py" --local "$@"

echo "=== MLP Model Training Completed ==="