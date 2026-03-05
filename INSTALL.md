# SMARTCS Pole AI - 설치 가이드

전주 파단 예측 AI 프로젝트의 환경 설정 가이드입니다.

## 📋 시스템 요구사항

### 기본 요구사항
- **Python**: 3.10+ (권장: 3.10.x)
- **운영체제**: Windows 10/11, Linux, macOS
- **메모리**: 8GB RAM 이상 (권장: 16GB+)
- **저장공간**: 10GB 이상 여유 공간

### GPU 환경 (선택사항)
- **GPU**: NVIDIA GPU (RTX 20xx 시리즈 이상 권장)
- **GPU 메모리**: 8GB 이상 권장
- **CUDA**: 11.8 또는 12.x
- **cuDNN**: 8.6+

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/earthscience2/SMARTCS_pole_ai.git
cd SMARTCS_pole_ai
```

### 2. Python 가상환경 생성

#### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS
```bash
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

#### 🖥️ CPU 환경 (기본)
```bash
pip install -r requirements.txt
```

#### 🚀 GPU 환경 (CUDA 지원)
```bash
# CUDA Toolkit과 cuDNN이 미리 설치되어 있어야 합니다
pip install -r requirements-gpu.txt
```

#### 👨‍💻 개발 환경
```bash
pip install -r requirements.txt -r requirements-dev.txt
# 또는 GPU 환경에서
pip install -r requirements-gpu.txt -r requirements-dev.txt
```

## 🔧 GPU 환경 설정 (Windows)

### 1. CUDA Toolkit 설치
1. [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) 방문
2. Windows > x86_64 > 11 > exe (local) 선택
3. 다운로드 후 설치 실행

### 2. cuDNN 설치
1. [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) 방문 (계정 필요)
2. CUDA 버전에 맞는 cuDNN 다운로드
3. 압축 해제 후 CUDA 설치 폴더에 복사

### 3. 환경변수 설정
```powershell
# 시스템 환경변수에 추가
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
```

### 4. GPU 인식 확인
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Count: ", len(tf.config.list_physical_devices('GPU')))
```

## 📁 프로젝트 구조

```
SMARTCS_pole_ai/
├── main/                          # 메인 파이프라인
│   ├── 1. make_data set/          # 데이터 생성
│   ├── 2. make_light_model/       # 라이트 모델
│   ├── 3. make_hard_model/        # 하드 모델
│   └── 4. make_mlp_model/         # MLP 모델
├── config/                        # 설정 파일
├── log/                          # 로그 파일
├── requirements.txt              # CPU 환경 패키지
├── requirements-gpu.txt          # GPU 환경 패키지
├── requirements-dev.txt          # 개발 환경 패키지
└── README.md                     # 프로젝트 설명
```

## 🗄️ 데이터베이스 설정

### MySQL 연결 설정
1. `config/poleconf.py` 파일 수정
2. 데이터베이스 연결 정보 입력:
   ```python
   poledb_host = 'your-host:port'
   poledb_dbname = 'your-database'
   poledb_user = 'your-username'
   poledb_pwd = 'your-password'
   ```

## 🧪 설치 확인

### 1. 기본 패키지 테스트
```python
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

print("✅ 모든 패키지가 정상적으로 설치되었습니다!")
```

### 2. GPU 테스트 (GPU 환경)
```python
import tensorflow as tf
print("GPU 사용 가능:", tf.test.is_gpu_available())
print("GPU 장치 목록:", tf.config.list_physical_devices('GPU'))
```

### 3. 데이터베이스 연결 테스트
```python
from config import poledb
poledb.poledb_init("main")  # 또는 "is", "kh"
poledb.ping()
```

## 🚨 문제 해결

### 일반적인 문제들

#### 1. TensorFlow GPU 인식 안됨
```bash
# CUDA/cuDNN 버전 확인
nvidia-smi
nvcc --version

# TensorFlow 재설치
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### 2. 메모리 부족 오류
- 배치 크기 줄이기
- GPU 메모리 증가 설정:
```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### 3. 패키지 충돌
```bash
# 가상환경 재생성
deactivate
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
python -m venv venv
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. Python 버전 (3.10+ 권장)
2. 가상환경 활성화 상태
3. CUDA/cuDNN 버전 호환성 (GPU 환경)
4. 시스템 메모리 및 GPU 메모리

추가 도움이 필요하면 GitHub Issues에 문의하세요.