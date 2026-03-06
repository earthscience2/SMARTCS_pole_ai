# 4. make_mlp_model

Light/Hard 모델 출력을 결합해 MLP 최종 판정 모델을 학습하는 단계입니다.

## 파일 구성

- `1. mlp_train_data.py`
  - Light/Hard 베스트 모델 출력으로 MLP 학습 데이터셋 생성
  - 출력: `1. mlp_train_data/<run>/`
- `2. mlp_model.py`
  - MLP 모델 학습/평가
  - 로컬 베스트 갱신: `best_model/`
  - 최종 베스트 모델 세트 동기화: `main/best_model/`
- `2. mlp_model_gpu.sh`
  - WSL2 GPU 환경에서 `mlp_model.py` 실행

## 실행 순서

```bash
python "main/4. make_mlp_model/1. mlp_train_data.py"
python "main/4. make_mlp_model/2. mlp_model.py" --local
```

또는 GPU 실행:

```bash
bash "main/4. make_mlp_model/2. mlp_model_gpu.sh"
```

## 주요 옵션

### `1. mlp_train_data.py`

- `--hard-data-run`: 사용할 hard 데이터 run 지정 (미지정 시 최신 run)
- `--output-dir`: 출력 디렉터리 (기본: `1. mlp_train_data`)
- `--run-name`: 출력 run 이름
- `--seed`: 시드
- `--device {auto,cpu,gpu}`: TensorFlow 디바이스

### `2. mlp_model.py`

- `--train-data-run`: 사용할 MLP train data run 지정 (미지정 시 최신 run)
- `--run-name`: 모델 run 이름

학습 서브 파라미터(은닉층, 정규화, 임계값 목표 등)는 CLI가 아니라 `USER_OPTIONS`에서 관리합니다.

## 로그 형식

- 주요 로그는 `[정보]`, `[경고]`, `[오류]` 접두어를 사용합니다.
