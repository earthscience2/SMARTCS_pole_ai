# AI 모델 파이프라인 가이드

이 문서는 `main/` 디렉토리 내 스크립트들의 실행 순서와 주요 CLI 옵션을 나열합니다.
각 스크립트의 최신 전체 옵션 목록은 `--help`를 사용하세요.

## 1) 데이터 수집 / 준비

### `1. get_project_info_list.py`
- 목적: 서버별 프로젝트 목록 및 분석 상태 수집
- 실행:
```bash
python "main/1. make_data set/1. get_project_info_list.py" [--server all|main|is|kh] [--output-dir "main/1. make_data set/1. project_info_list"]
```

### `2. get_anal_pole_list.py`
- 목적: 분석된 전주 목록 수집
- 실행:
```bash
python "main/1. make_data set/2. get_anal_pole_list.py" \
  [--server all|main|is|kh] \
  [--project-list-json "<path>"] \
  [--project-list-dir "main/1. make_data set/1. project_info_list"] \
  [--output-dir "main/1. make_data set/2. anal_pole_list"]
```

### `3. get_raw_pole_data.py`
- 목적: 축별 원시 측정 CSV 파일 다운로드
- 실행:
```bash
python "main/1. make_data set/3. get_raw_pole_data.py" \
  [--input-json "<path>"] \
  [--input-dir "main/1. make_data set/2. anal_pole_list"] \
  [--output-dir "main/1. make_data set/3. raw_pole_data"] \
  [--normal-ratio 10]
```

### `3.1. check_raw_pole_data_info.py`
- 목적: 원시 데이터 통계 요약
- 실행:
```bash
python "main/1. make_data set/3.1. check_raw_pole_data_info.py" [--no-plot]
```

### `4. merge_data.py`
- 목적: x/y/z CSV 데이터를 처리된 그리드 CSV로 병합
- 실행:
```bash
python "main/1. make_data set/4. merge_data.py" \
  [--raw-data-dir "3. raw_pole_data"] \
  [--output-dir "4. merge_data"] \
  [--normal-ratio 10]
```

### `4.1. check_merge_data_info.py`
- 목적: 병합된 데이터 통계 요약
- 실행:
```bash
python "main/1. make_data set/4.1. check_merge_data_info.py" [--data-dir "main/1. make_data set/4. merge_data"] [--no-plot]
```

### `5. edit_data.py`
- 목적: ROI/파단 영역 편집 GUI
- 실행:
```bash
python "main/1. make_data set/5. edit_data.py" [--input-dir "main/1. make_data set/4. merge_data/break"]
```

### `6. set_light_train_data.py`
- 목적: 라이트 모델 훈련/테스트 NPY 데이터셋 구축
- 실행:
```bash
python "main/1. make_data set/6. set_light_train_data.py" \
  [--data-dir "4. merge_data"] \
  [--output-dir "6. light_train_data"] \
  [--break-data-dir "5. edit_data"] \
  [--sort-by height|degree] \
  [--min-points 200] \
  [--max-points 400] \
  [--run-subdir "<run_name>"]
```

## 2) 라이트 모델

### `1. make_light_model.py`
- 목적: 라이트 모델 훈련
- 실행:
```bash
python "main/2. make_light_model/1. make_light_model.py" \
  [--epochs 120] [--batch-size 32] [--learning-rate 0.001] \
  [--focal-alpha 0.95] [--break-class-weight-scale 1.45] \
  [--run-tag "<tag>"]
```

### `1. make_light_model_wsl2.sh`
- 목적: 라이트 모델 루프용 WSL2 래퍼
- 실행:
```bash
bash "main/2. make_light_model/1. make_light_model_wsl2.sh"
```

### `2. evaluate_light_model.py`
- 목적: 라이트 모델 평가 및 최적 모델 업데이트
- 실행:
```bash
python "main/2. make_light_model/2. evaluate_light_model.py" \
  [--run "<light_run_name>"] \
  [--target-precision 0.80] \
  [--target-recall 0.90] \
  [--target-f1 0.84] \
  [--target-pass-mode all_metrics|any_metric] \
  [--best-target-recall 0.90] \
  [--best-target-accuracy 0.50]
```

## 3) 하드 모델 (1단계)

### `1. set_hard_train_data.py`
- 목적: 하드 모델 훈련 데이터셋 구축
- 실행:
```bash
python "main/3. make_hard_model/1. set_hard_train_data.py" \
  [--data-dir "4. merge_data"] \
  [--output-dir "9. hard_train_data"] \
  [--roi-dir "5. edit_data"] \
  [--run-subdir "<run_name>"] \
  [--sort-by height|degree] \
  [--min-points 200] \
  [--max-points 400]
```

### `2. make_hard_model_1st.py`
- 목적: 하드 1단계 bbox 모델 훈련
- 실행:
```bash
python "main/3. make_hard_model/2. make_hard_model_1st.py" \
  [--exp 0] [--epochs 300] [--batch-size 32] \
  [--learning-rate 0.001] [--dropout 0.3] [--run-tag "<tag>"]
```

### `2. make_hard_model_1st_wsl2.sh`
- 목적: 하드1 루프용 WSL2 래퍼
- 실행:
```bash
bash "main/3. make_hard_model/2. make_hard_model_1st_wsl2.sh"
```

### `2. evaluate_hard_model_1st.py`
- 목적: 하드 1단계 bbox 모델 평가
- 실행:
```bash
python "main/3. make_hard_model/2. evaluate_hard_model_1st.py" \
  [--run_dir "<run_path>"] \
  [--target-mean-best-iou 0.45] \
  [--target-ratio-iou-0-5 0.55] \
  [--target-ratio-iou-0-7 0.30] \
  [--target-pass-mode all_axes|average]
```

### `2. evaluate_hard_model_1st_wsl2.sh`
- 목적: 단일 하드1 평가용 WSL2 래퍼
- 실행:
```bash
bash "main/3. make_hard_model/2. evaluate_hard_model_1st_wsl2.sh"
```

## 4) 하드 모델 (2단계)

### `3. make_hard_model_2nd.py`
- 목적: 하드 2단계 신뢰도 모델 훈련
- 실행:
```bash
python "main/3. make_hard_model/3. make_hard_model_2nd.py" \
  [--first-stage-run-dir "main/3. make_hard_model/hard_model_1st_best"] \
  [--batch-size 32] [--epochs 220] [--learning-rate 0.001] \
  [--run-tag "<tag>"]
```

### `3. make_hard_model_2nd_wsl2.sh`
- 목적: 하드2 루프용 WSL2 래퍼
- 실행:
```bash
bash "main/3. make_hard_model/3. make_hard_model_2nd_wsl2.sh"
```

### `3. evaluate_hard_model_2nd.py`
- 목적: 하드 2단계 신뢰도 모델 평가
- 실행:
```bash
python "main/3. make_hard_model/3. evaluate_hard_model_2nd.py" \
  [--data-dir "4. merge_data"] [--run-dir "<run_path>"] \
  [--min-points 200] [--max-points 400] [--max-files 0] \
  [--only-break] [--only-normal] [--batch-size 128] [--repreprocess] \
  [--confidence-threshold 0.0] [--min-box-ratio 0.02] \
  [--min-box-degree-span 5.0] [--min-box-height-span 0.1] \
  [--test-only | --no-test-only] \
  [--target-overall-best-f1 0.70] [--target-overall-auc 0.70] \
  [--target-overall-separation 0.20] \
  [--target-pass-mode all_metrics|any_metric]
```

### `3. evaluate_hard_model_2nd_wsl2.sh`
- 목적: 단일 하드2 평가용 WSL2 래퍼
- 실행:
```bash
bash "main/3. make_hard_model/3. evaluate_hard_model_2nd_wsl2.sh"
```

## 5) 최종 결정 / 배포 / 실제 추론

### `1. mlp_train_data.py`
- 목적: MLP 훈련 데이터셋 생성
- 실행:
```bash
python "main/4. make_mlp_model/1. mlp_train_data.py" \
  [--light-model-dir "best_light_model"] \
  [--hard1-model-dir "best_hard_model_1st"] \
  [--hard2-model-dir "best_hard_model_2nd"] \
  [--weight-light 0.4] [--weight-hard1 0.3] [--weight-hard2 0.3] \
  [--weight-x 0.34] [--weight-y 0.33] [--weight-z 0.33] \
  [--no-save]
```

### `2. mlp_model.py`
- 목적: 최적 라이트 + 하드 결과 결합 및 최종 MLP 결정 모델 훈련
- 실행:
```bash
python "main/4. make_mlp_model/2. mlp_model.py" \
  [--train-data-run "<run_name>"] \
  [--light-model-dir "best_light_model"] \
  [--hard1-model-dir "best_hard_model_1st"] \
  [--hard2-model-dir "best_hard_model_2nd"] \
  [--run-name "<name>"] [--seed 42] \
  [--mlp-hidden-layers "32,16"] [--mlp-alpha 1e-4] [--mlp-max-iter 1200] [--val-size 0.3] \
  [--weight-light 0.4] [--weight-hard1 0.3] [--weight-hard2 0.3] \
  [--weight-x 0.34] [--weight-y 0.33] [--weight-z 0.33] \
  [--target-suspect-recall 0.90] [--target-break-precision 0.90]
```

### `2. mlp_model_gpu.sh`
- 목적: MLP 모델 GPU 훈련용 WSL2 래퍼
- 실행:
```bash
bash "main/4. make_mlp_model/2. mlp_model_gpu.sh"
```

### `make_save_model.py`
- 목적: 라이트/하드2/mlp 배포 가능한 패키지 및 테스트 아티팩트 내보내기
- 실행:
```bash
python "main/make_save_model.py" \
  [--light-dir "best_light_model"] \
  [--hard2-dir "best_hard_model_2nd"] \
  [--mlp-run "<mlp_run_name>"] \
  [--mlp-base-dir "4. make_mlp_model"] \
  [--output-dir "make_save_model"] \
  [--package-name "<name>"]
```

### `test_ai.py`
- 목적: DB 가져오기부터 Excel 예측까지의 프로덕션 스타일 전체 파이프라인
- 출력: `by_measure`, `by_pole` 시트가 있는 Excel
- 실행:
```bash
python "main/test_ai.py" \
  --server main|is|kh \
  [--project "<project_name>"] [--pole-id "<pole_id>"] \
  [--max-poles 0] [--min-points 200] [--max-points 400] \
  [--saved-package "<package_name>"] [--saved-base-dir "make_save_model"] \
  [--output-dir "test_ai"]
```

## 6) 유틸리티

### `check_text_encoding.py`
- 목적: 파일의 텍스트 인코딩 문제 확인
- 실행:
```bash
python "main/check_text_encoding.py"
```

### `export_hard_2nd_conf_to_savedmodel.py`
- 목적: 하드2 `.keras` 신뢰도 모델을 SavedModel 형식으로 내보내기
- 실행:
```bash
python "main/export_hard_2nd_conf_to_savedmodel.py" \
  [--source-dir "<hard2_run_or_best_dir>"] \
  [--output-dir "<output_dir>"] \
  [--local]
```

### `export_hard_2nd_conf_to_savedmodel_wsl2.sh`
- 목적: 하드2 SavedModel 내보내기용 WSL2 래퍼
- 실행:
```bash
bash "main/export_hard_2nd_conf_to_savedmodel_wsl2.sh"
```

### `hard_model_2nd_savedmodel_export.md`
- 목적: 하드2 SavedModel 내보내기 상세 노트

### `plot_processed_csv_2d.py`
- 목적: 처리된 CSV를 2D 등고선/이미지로 플롯
- 사용법:
```python
from main.plot_processed_csv_2d import plot_csv_2d
plot_csv_2d("sample.csv")
```

## 권장 실행 순서

1. `1` -> `2` -> `3` -> `3.1`
2. `4` -> `4.1` -> `5` -> `6`
3. `라이트 모델 1` -> `라이트 모델 2` (목표 성능까지 반복)
4. `하드 모델 1` -> `하드 모델 2` -> `하드 모델 2 평가`
5. `하드 모델 3` -> `하드 모델 3 평가`
6. `MLP 1` -> `MLP 2` -> `배포`

