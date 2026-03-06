# 2. make_light_model

라이트 모델 데이터 생성, 학습, 평가, 베스트 모델 관리 폴더입니다.

## 폴더 구성

- `1. set_light_train_data.py`: 학습용 NPY 데이터셋 생성
- `2. make_light_model.py`: 단일 라이트 모델 학습 + 평가 + 베스트 모델 갱신
- `2. make_light_model_gpu.sh`: WSL2 GPU 환경에서 반복 학습/평가 루프 실행
- `1. light_train_data/`: 생성된 학습 데이터셋 run 저장소
- `2. light_models/`: 학습된 모델 run 저장소
- `best_light_model/`: 현재 베스트 모델 별칭 폴더 (`best_light_model/<run>/` 구조)

## 전체 흐름

1. `main/1. make_data set` 단계 완료 (`4. merge_data`, `5. edit_data` 준비)
2. `1. set_light_train_data.py` 실행해서 NPY 생성
3. `2. make_light_model.py` 실행해서 모델 학습/평가
4. 최고 모델은 자동으로 `best_light_model/<run>/` 형태로 갱신

## 1) 학습 데이터 생성

```bash
python "main/2. make_light_model/1. set_light_train_data.py"
```

주요 옵션:

- `--data-dir` 기본 `4. merge_data`
- `--break-data-dir` 기본 `5. edit_data` (ROI 반영 파단 데이터)
- `--output-dir` 기본 `1. light_train_data`
- `--min-points` 기본 `200`
- `--max-points` 기본 `400`
- `--run-subdir` 미지정 시 타임스탬프 폴더 생성

출력 예:

- `1. light_train_data/<run>/train/break_imgs_train.npy`
- `1. light_train_data/<run>/test/break_imgs_test.npy`
- `1. light_train_data/<run>/break_imgs_metadata.json`

## 2) 단일 모델 학습/평가

```bash
python "main/2. make_light_model/2. make_light_model.py" --local
```

Windows에서 `--local` 없이 실행하면 WSL2 `.sh`를 호출합니다.

### 빠른 설정 방법

`2. make_light_model.py` 상단 `USER_OPTIONS`에서 기본값을 한 번에 수정할 수 있습니다.

주요 항목:

- `epochs`, `batch_size`, `learning_rate`
- `focal_alpha`, `break_class_weight_scale`
- `fixed_threshold`
- `target_precision`, `target_recall`, `target_f1`

run 폴더명 규칙:

- 항상 `YYYYMMDD_HHMM` 형식만 사용
- 예: `20260212_1423`

출력 예:

- `2. light_models/<run>/checkpoints/best.keras`
- `2. light_models/<run>/results/training_config.json`
- `2. light_models/<run>/evaluation/evaluation_report.json`

학습 로그:

- 상세 평가 시작 후 콘솔에 `[정보][평가 1/7]` ~ `[정보][평가 7/7]` 로그 출력
- 베스트 모델 교체/유지/건너뜀 결과를 한국어 로그로 출력

## 3) WSL2 GPU 루프 학습

```bash
bash "main/2. make_light_model/2. make_light_model_gpu.sh"
```

환경 변수로 루프 설정 가능:

- `LIGHT_MAX_ROUNDS` (기본: 1, 예: `LIGHT_MAX_ROUNDS=3`)
- `LIGHT_EPOCHS`, `LIGHT_EPOCHS_STEP`
- `LIGHT_LR`, `LIGHT_LR_DECAY`
- `LIGHT_FOCAL_ALPHA_SCHEDULE`
- `LIGHT_BREAK_WEIGHT_SCALE_SCHEDULE`

가상환경 우선순위:

1. `venv_gpu`
2. `venv_wsl2`

## 베스트 모델 관리

`2. make_light_model.py`는 평가 후 `2. light_models` 전체 리포트를 비교해 최고 모델이면:

- `best_light_model/<run>/` 교체 저장 (run 폴더명 유지)
- `best_light_model/best_model_selection.json` 기록
- `2. light_models/best_model_selection_history.jsonl` 이력 누적

## 자주 발생하는 문제

- `ModuleNotFoundError: No module named 'tensorflow'`
  - 원인: WSL2 Python 환경에 tensorflow 미설치
  - 조치:
    - `source venv_gpu/bin/activate` 또는 `source venv_wsl2/bin/activate`
    - `pip install tensorflow` (GPU 환경이면 `tensorflow[and-cuda]` 검토)

- `1. light_train_data` run을 못 찾는 경우
  - `1. set_light_train_data.py`를 먼저 실행해 `train/`, `test/` NPY 생성 필요

## 권장 실행 순서

```bash
python "main/2. make_light_model/1. set_light_train_data.py"
python "main/2. make_light_model/2. make_light_model.py" --local
```

또는 루프 실행:

```bash
bash "main/2. make_light_model/2. make_light_model_gpu.sh"
```
