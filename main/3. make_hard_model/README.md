# make_hard_model

Hard 모델용 데이터 생성, 1차 모델 학습, 2차 모델 학습을 관리하는 디렉터리입니다.

현재 2차 모델은 `1차 모델별 베스트`와 `전체 베스트`를 함께 관리합니다.

## 파일 구성

- `1. set_hard_train_data.py`
  - CSV를 읽어 Hard 학습용 NPY를 생성
  - 출력: `1. hard_train_data/<run>/train`, `1. hard_train_data/<run>/test`
- `2. make_hard_model_1st.py`
  - Hard 1차 bbox 모델 학습
  - 학습 후 평가와 `best_hard_model_1st` 갱신 포함
- `2. make_hard_model_1st_gpu.sh`
  - WSL2/GPU에서 1차 모델 학습 실행
- `3. make_hard_model_2nd.py`
  - Hard 2차 conf head 모델 학습
  - 학습 후 평가, 1차 모델별 베스트 선정, 전체 베스트 선정 포함
- `3. make_hard_model_2nd_gpu.sh`
  - WSL2/GPU에서 2차 모델 학습 실행
- `show_best_hard2_models.py`
  - 1차 모델별 베스트 2차 모델과 전체 베스트 2차 모델을 출력

## 2차 모델 동작 방식

`3. make_hard_model_2nd.py`는 항상 특정 1차 모델을 기준으로 학습됩니다.

- 기본값: `best_hard_model_1st`가 가리키는 1차 모델 사용
- 선택 실행: `--first-stage-model <run_name>` 또는 `HARD2_FIRST_STAGE_MODEL`
- 조회만: `--list-first-stage-models`

학습이 끝나면 아래 두 가지가 함께 갱신됩니다.

- `best_hard_model_2nd/by_first_stage/<1차_run>/`
  - 해당 1차 모델을 기준으로 가장 좋은 2차 모델
- `best_hard_model_2nd/overall_best/`
  - 모든 1차 모델별 베스트 중에서 가장 좋은 2차 모델

기존 단일 구조의 `best_hard_model_2nd/<run>/`를 사용하던 경우, 스크립트가 필요 시 새 구조로 마이그레이션합니다.

## 출력 경로

### 1차

- run: `2. hard_models_1st/<run_name>`
- 평가: `2. hard_models_1st/<run_name>/evaluate/`
- best alias: `best_hard_model_1st/<선정_run>/`
- history: `2. hard_models_1st/best_model_selection_history.jsonl`

### 2차

- run: `3. hard_models_2nd/<YYYYMMDD_HHMM>`
- 평가: `3. hard_models_2nd/<run>/evaluate/`
- 1차별 best alias: `best_hard_model_2nd/by_first_stage/<1차_run>/<2차_run>/`
- 전체 best alias: `best_hard_model_2nd/overall_best/<2차_run>/`
- history: `3. hard_models_2nd/best_model_selection_history.jsonl`

## 주요 옵션

### 1차 `2. make_hard_model_1st.py`

상단 `USER_OPTIONS`에서 관리합니다.

- `core`
  - `epochs`, `batch_size`, `learning_rate`, `dropout`, `pred_boxes_per_axis`
- `sub`
  - `use_augmentation`
  - `conf_weight`
  - `iou_loss_weight`, `anchor_reg_weight`
  - `target_mean_best_iou`, `target_ratio_iou_0_5`, `target_ratio_iou_0_7`, `target_pass_mode`

### 2차 `3. make_hard_model_2nd.py`

상단 `USER_OPTIONS`에서 관리합니다.

- `epochs`, `batch_size`, `learning_rate`
- `target_overall_best_f1`, `target_overall_auc`, `target_overall_separation`, `target_pass_mode`

CLI 옵션:

- `--first-stage-run-dir`
- `--first-stage-model`
- `--list-first-stage-models`
- `--batch-size`
- `--epochs`
- `--learning-rate`
- `--cpu`
- `--local`

## 실행 예시

### Hard 데이터 생성

```bash
python "main/3. make_hard_model/1. set_hard_train_data.py"
```

### 1차 학습

```bash
bash "main/3. make_hard_model/2. make_hard_model_1st_gpu.sh"
```

### 2차 학습

```bash
bash "main/3. make_hard_model/3. make_hard_model_2nd_gpu.sh"
```

주요 환경변수:

- `HARD2_FIRST_STAGE_RUN_DIR`
- `HARD2_FIRST_STAGE_MODEL`
- `HARD2_EPOCHS`
- `HARD2_BATCH_SIZE`
- `HARD2_LR`

### 특정 1차 모델 기준으로 2차 학습

```bash
export HARD2_FIRST_STAGE_MODEL="20260305_1327"
bash "main/3. make_hard_model/3. make_hard_model_2nd_gpu.sh"
```

또는

```bash
python "main/3. make_hard_model/3. make_hard_model_2nd.py" --first-stage-model "20260305_1327"
```

### 사용 가능한 1차 모델 목록 확인

```bash
python "main/3. make_hard_model/3. make_hard_model_2nd.py" --list-first-stage-models
```

## 후속 단계 연동

`main/4. make_mlp_model/1. mlp_train_data.py`는 이제 선택된 Hard1 모델에 맞는 Hard2 모델을
`best_hard_model_2nd/by_first_stage/<hard1_run>/`에서 자동으로 읽습니다.

즉, Hard1과 Hard2가 서로 다른 계열 run으로 섞이지 않도록 맞춰집니다.
