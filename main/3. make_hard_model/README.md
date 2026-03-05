# make_hard_model

Hard 모델 데이터 생성/학습 폴더입니다.
현재 구조는 **1차/2차 학습 스크립트 내부에 평가 + 베스트 선정 로직이 통합**되어 있습니다.

## 파일 구성

- `1. set_hard_train_data.py`
  - CSV를 읽어 Hard 학습용 NPY 생성
  - 출력: `1. hard_train_data/<run>/train,test`
  - `4. merge_data`, `5. edit_data`는 `main/1. make_data set` 하위 자동 탐색
- `2. make_hard_model_1st.py`
  - Hard 1차(bbox) 학습
  - 학습 후 통합 평가 + best_hard_model_1st 갱신
- `2. make_hard_model_1st_gpu.sh`
  - WSL2/GPU에서 1차 학습 실행
- `3. make_hard_model_2nd.py`
  - Hard 2차(conf head) 학습
  - 학습 후 통합 평가 + best_hard_model_2nd 갱신
- `3. make_hard_model_2nd_gpu.sh`
  - WSL2/GPU에서 2차 학습 실행

## 주요 변경사항 반영

- 분리 평가 스크립트(`evaluate_hard_model_*.py`) 삭제 후 학습 코드 내부 통합
- 평가 결과는 각 run 폴더 내부로 저장
- 베스트 모델은 run 폴더명 그대로 alias 폴더로 복사
- 중복 파라미터 실행 감지 시 학습/평가 스킵
- 1차 `conf_weight`는 고정 `0.0`
- 1차 옵션은 `USER_OPTIONS`를 `core` / `sub`로 분리

## 출력 경로

### 1차

- run: `2. hard_models_1st/<run_name>`
- 평가: `2. hard_models_1st/<run_name>/evaluate/`
  - `evaluation_metrics.json`
  - `training_feedback.json`
- best alias: `best_hard_model_1st/<선정_run_폴더명>/`
- history: `2. hard_models_1st/best_model_selection_history.jsonl`

### 2차

- run: `3. hard_models_2nd/<YYYYMMDD_HHMM>`
- 평가: `3. hard_models_2nd/<run>/evaluate/`
  - (신규 run) `evaluation_metrics.json`, `training_feedback.json`
  - (구버전 run 이동분) `test/`, `all/` 하위 구조가 남아있을 수 있음
- best alias: `best_hard_model_2nd/<선정_run_폴더명>/`
- history: `3. hard_models_2nd/best_model_selection_history.jsonl`

## 옵션 설정 방법

### 1차(`2. make_hard_model_1st.py`)

코드 상단 `USER_OPTIONS`에서 직접 수정:

- `core`
  - `epochs`, `batch_size`, `learning_rate`, `dropout`, `pred_boxes_per_axis`
- `sub`
  - `use_augmentation`
  - `conf_weight` (고정값, 현재 0.0)
  - `iou_loss_weight`, `anchor_reg_weight`
  - `target_mean_best_iou`, `target_ratio_iou_0_5`, `target_ratio_iou_0_7`, `target_pass_mode`

### 2차(`3. make_hard_model_2nd.py`)

코드 상단 `USER_OPTIONS`에서 직접 수정:

- `epochs`, `batch_size`, `learning_rate`
- `target_overall_best_f1`, `target_overall_auc`, `target_overall_separation`, `target_pass_mode`

## 중복 파라미터 감지

동일한 핵심 파라미터 조합으로 이미 학습된 run이 있으면 아래 메시지 후 종료합니다.

- `중복 파라미터 감지: 기존 run '...'와 동일한 학습 설정입니다. 이번 실행은 학습/평가를 건너뜁니다.`

## 실행

### 데이터 생성

```bash
python "main/3. make_hard_model/1. set_hard_train_data.py"
```

### 1차 학습 (WSL2 GPU)

```bash
bash "main/3. make_hard_model/2. make_hard_model_1st_gpu.sh"
```

주요 env:

- `HARD1_EPOCHS`
- `HARD1_BATCH_SIZE`
- `HARD1_TARGET_MEAN_IOU`
- `HARD1_TARGET_IOU_05`
- `HARD1_TARGET_IOU_07`
- `HARD1_PASS_MODE`

### 2차 학습 (WSL2 GPU)

```bash
bash "main/3. make_hard_model/3. make_hard_model_2nd_gpu.sh"
```

주요 env:

- `HARD2_FIRST_STAGE_RUN_DIR` (기본: best_hard_model_1st)
- `HARD2_FIRST_STAGE_MODEL` (특정 1차 모델 선택, 예: "20260305_1327")
- `HARD2_EPOCHS`
- `HARD2_BATCH_SIZE`
- `HARD2_LR`
- `HARD2_TARGET_OVERALL_BEST_F1`
- `HARD2_TARGET_OVERALL_AUC`
- `HARD2_TARGET_OVERALL_SEPARATION`
- `HARD2_PASS_MODE`

### 특정 1차 모델 선택하여 2차 학습

기본적으로 `best_hard_model_1st`를 사용하지만, 특정 1차 모델을 선택할 수 있습니다:

```bash
# 환경변수로 설정
export HARD2_FIRST_STAGE_MODEL="20260305_1327"
bash "main/3. make_hard_model/3. make_hard_model_2nd_gpu.sh"

# 또는 직접 Python 스크립트 실행
python "main/3. make_hard_model/3. make_hard_model_2nd.py" --first-stage-model "20260305_1327"
```

## 참고

- WSL 가상환경(`venv_gpu` 또는 `venv_wsl2`)에 TensorFlow 설치 필요
- 2차 학습 전 1차 best 체크포인트(`best_x/y/z.keras`) 필요
