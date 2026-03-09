# 4. make_mlp_model

Light / Hard1 / Hard2 모델의 출력을 결합해 최종 MLP 판단 모델을 만드는 단계입니다.

## 파일 구성

- `1. mlp_train_data.py`
  - Light, Hard1, Hard2 베스트 모델 출력으로 MLP 학습 데이터를 생성
  - 출력: `1. mlp_train_data/<run>/`
- `2. mlp_model.py`
  - MLP 모델 학습 및 평가
  - 로컬 베스트 갱신: `best_model/by_dependency/...`, `best_model/overall_best/`
  - 최종 베스트 묶음 갱신: `main/best_model/`
- `2. mlp_model_gpu.sh`
  - WSL2/GPU 환경에서 `mlp_model.py` 실행

## 중요한 연동 변경

Hard2 베스트 구조가 아래처럼 바뀌었습니다.

- `best_hard_model_2nd/by_first_stage/<hard1_run>/...`
- `best_hard_model_2nd/overall_best/...`

그래서 `1. mlp_train_data.py`는 이제:

- 먼저 `best_hard_model_1st`에서 선택된 Hard1 모델을 읽고
- 그 Hard1 run과 연결된 Hard2 베스트를 `by_first_stage/<hard1_run>/`에서 자동 선택합니다.

즉, MLP 데이터 생성 시 Hard1과 Hard2가 서로 다른 1차 계열 run으로 섞이지 않습니다.

또한 `2. mlp_model.py`는 저장된 메타데이터를 기준으로 같은 Hard2 원본 run을 다시 찾아
`main/best_model/hard2_model/`로 복사합니다.

추가로 MLP 베스트도 이제 아래처럼 관리합니다.

- `best_model/by_dependency/<light+hard1+hard2 조합>/`
  - 같은 하위 모델 조합으로 학습한 MLP들끼리만 비교한 베스트
- `best_model/overall_best/`
  - 모든 조합별 베스트 중 전체 최고 성능 MLP

`main/best_model/`은 항상 `overall_best` 기준으로만 갱신됩니다.

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

- `--hard-data-run`
- `--light-model-dir`
- `--hard1-model-dir`
- `--hard2-model-dir`
- `--output-dir`
- `--run-name`
- `--seed`
- `--device {auto,cpu,gpu}`

기본 `--hard2-model-dir`는 `best_hard_model_2nd`이며,
내부에서 Hard1 선택 결과에 맞는 `by_first_stage/<hard1_run>/`를 자동 해석합니다.

### `2. mlp_model.py`

- `--train-data-run`
- `--run-name`
- `--hidden-layers 64,32,16`
- `--alpha 0.0001`
- `--max-iter 1200`
- `--early-stopping` / `--no-early-stopping`
- `--validation-fraction 0.2`
- `--n-iter-no-change 25`
- `--seed 42`
- `--target-suspect-recall 0.90`
- `--target-break-precision 0.90`
- `--target-binary-alert-f1 0.75`
- `--target-binary-break-f1 0.70`
- `--target-binary-break-auc 0.80`
- `--target-pass-mode all_metrics`

기본값은 `USER_OPTIONS`에 있고, 실행 시 CLI 옵션으로 덮어쓸 수 있습니다.

예시:

```bash
python "main/4. make_mlp_model/2. mlp_model.py" \
  --train-data-run "20260310_002231" \
  --hidden-layers 128,64,32 \
  --alpha 0.0005 \
  --max-iter 1500 \
  --target-binary-alert-f1 0.73 \
  --target-binary-break-f1 0.78
```

## 생성 결과

### MLP 학습 데이터

- `1. mlp_train_data/<run>/`
  - feature / split 데이터
  - 메타데이터
  - 사용한 Light / Hard1 / Hard2 모델 정보

### 최종 베스트 모델 묶음

- `main/best_model/light_model/`
- `main/best_model/hard1_model/`
- `main/best_model/hard2_model/`
- `main/best_model/mlp_model/`
- `main/best_model/model_dependency_info.json`

### 로컬 MLP 베스트 구조

- `main/4. make_mlp_model/best_model/by_dependency/<조합명>/`
  - `mlp_pipeline.joblib`
  - `best_model_selection.json`
- `main/4. make_mlp_model/best_model/overall_best/`
  - `mlp_pipeline.joblib`
  - `best_model_selection.json`
- `main/4. make_mlp_model/best_model/model_update_history.jsonl`

## 참고

- Hard2를 수동으로 직접 고르기보다, 먼저 Hard1 베스트를 확정한 뒤 MLP 데이터를 다시 만드는 흐름을 권장합니다.
- Hard2 메타데이터에는 `first_stage_run` 정보가 포함되어 있어 후속 복사 단계에서 같은 계열 run을 재사용합니다.
