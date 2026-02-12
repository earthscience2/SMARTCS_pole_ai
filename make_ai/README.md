# make_ai Pipeline Guide

This document lists the execution order and main CLI options for scripts in `make_ai/`.
Use each script's `--help` for the latest full option list.

## 1) Data Collection / Preparation

### `1. get_project_info_list.py`
- Purpose: collect project list and analysis status by server
- Run:
```bash
python "make_ai/1. get_project_info_list.py" [--server all|main|is|kh] [--output-dir "make_ai/1. project_info_list"]
```

### `2. get_anal_pole_list.py`
- Purpose: collect analyzed pole list
- Run:
```bash
python "make_ai/2. get_anal_pole_list.py" \
  [--server all|main|is|kh] \
  [--project-list-json "<path>"] \
  [--project-list-dir "make_ai/1. project_info_list"] \
  [--output-dir "make_ai/2. anal_pole_list"]
```

### `3. get_raw_pole_data.py`
- Purpose: download raw per-axis measurement CSV files
- Run:
```bash
python "make_ai/3. get_raw_pole_data.py" \
  [--input-json "<path>"] \
  [--input-dir "make_ai/2. anal_pole_list"] \
  [--output-dir "make_ai/3. raw_pole_data"] \
  [--normal-ratio 10]
```

### `3.1. check_raw_pole_data_info.py`
- Purpose: summarize raw data stats
- Run:
```bash
python "make_ai/3.1. check_raw_pole_data_info.py" [--no-plot]
```

### `4. merge_data.py`
- Purpose: merge x/y/z CSV data into processed grid CSV
- Run:
```bash
python "make_ai/4. merge_data.py" \
  [--raw-data-dir "3. raw_pole_data"] \
  [--output-dir "4. merge_data"] \
  [--normal-ratio 10]
```

### `4.1. check_merge_data_info.py`
- Purpose: summarize merged data stats
- Run:
```bash
python "make_ai/4.1. check_merge_data_info.py" [--data-dir "make_ai/4. merge_data"] [--no-plot]
```

### `5. edit_data.py`
- Purpose: ROI/break region editing GUI
- Run:
```bash
python "make_ai/5. edit_data.py" [--input-dir "make_ai/4. merge_data/break"]
```

### `6. set_light_train_data.py`
- Purpose: build light-model train/test NPY dataset
- Run:
```bash
python "make_ai/6. set_light_train_data.py" \
  [--data-dir "4. merge_data"] \
  [--output-dir "6. light_train_data"] \
  [--break-data-dir "5. edit_data"] \
  [--sort-by height|degree] \
  [--min-points 200] \
  [--max-points 400] \
  [--run-subdir "<run_name>"]
```

## 2) Light Model

### `7. make_light_model.py`
- Purpose: train light model
- Run:
```bash
python "make_ai/7. make_light_model.py" \
  [--epochs 120] [--batch-size 32] [--learning-rate 0.001] \
  [--focal-alpha 0.95] [--break-class-weight-scale 1.45] \
  [--run-tag "<tag>"]
```

### `7. make_light_model_wsl2.sh`
- Purpose: WSL2 wrapper for light model loop
- Run:
```bash
bash "make_ai/7. make_light_model_wsl2.sh"
```

### `8. evaluate_light_model.py`
- Purpose: evaluate light model and update best model
- Run:
```bash
python "make_ai/8. evaluate_light_model.py" \
  [--run "<light_run_name>"] \
  [--target-precision 0.80] \
  [--target-recall 0.90] \
  [--target-f1 0.84] \
  [--target-pass-mode all_metrics|any_metric] \
  [--best-target-recall 0.90] \
  [--best-target-accuracy 0.50]
```

## 3) Hard Model (1st Stage)

### `9. set_hard_train_data.py`
- Purpose: build hard-model training dataset
- Run:
```bash
python "make_ai/9. set_hard_train_data.py" \
  [--data-dir "4. merge_data"] \
  [--output-dir "9. hard_train_data"] \
  [--roi-dir "5. edit_data"] \
  [--run-subdir "<run_name>"] \
  [--sort-by height|degree] \
  [--min-points 200] \
  [--max-points 400]
```

### `10. make_hard_model_1st.py`
- Purpose: train hard 1st-stage bbox model
- Run:
```bash
python "make_ai/10. make_hard_model_1st.py" \
  [--exp 0] [--epochs 300] [--batch-size 32] \
  [--learning-rate 0.001] [--dropout 0.3] [--run-tag "<tag>"]
```

### `10. make_hard_model_1st_wsl2.sh`
- Purpose: WSL2 wrapper for hard1 loop
- Run:
```bash
bash "make_ai/10. make_hard_model_1st_wsl2.sh"
```

### `11. evaluate_hard_model_1st.py`
- Purpose: evaluate hard 1st-stage bbox model
- Run:
```bash
python "make_ai/11. evaluate_hard_model_1st.py" \
  [--run_dir "<run_path>"] \
  [--target-mean-best-iou 0.45] \
  [--target-ratio-iou-0-5 0.55] \
  [--target-ratio-iou-0-7 0.30] \
  [--target-pass-mode all_axes|average]
```

### `11. evaluate_hard_model_1st_wsl2.sh`
- Purpose: WSL2 wrapper for single hard1 evaluation
- Run:
```bash
bash "make_ai/11. evaluate_hard_model_1st_wsl2.sh"
```

## 4) Hard Model (2nd Stage)

### `12. make_hard_model_2nd.py`
- Purpose: train hard 2nd-stage confidence model
- Run:
```bash
python "make_ai/12. make_hard_model_2nd.py" \
  [--first-stage-run-dir "make_ai/hard_model_1st_best"] \
  [--batch-size 32] [--epochs 220] [--learning-rate 0.001] \
  [--run-tag "<tag>"]
```

### `12. make_hard_model_2nd_wsl2.sh`
- Purpose: WSL2 wrapper for hard2 loop
- Run:
```bash
bash "make_ai/12. make_hard_model_2nd_wsl2.sh"
```

### `13. evaluate_hard_model_2nd.py`
- Purpose: evaluate hard 2nd-stage confidence model
- Run:
```bash
python "make_ai/13. evaluate_hard_model_2nd.py" \
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

### `13. evaluate_hard_model_2nd_wsl2.sh`
- Purpose: WSL2 wrapper for single hard2 evaluation
- Run:
```bash
bash "make_ai/13. evaluate_hard_model_2nd_wsl2.sh"
```

## 5) Final Decision / Deployment / Real Inference

### `14. best_model_result_and_mlp_final.py`
- Purpose: combine best light + hard2 outputs and train final MLP decision model
- Run:
```bash
python "make_ai/14. best_model_result_and_mlp_final.py" \
  [--hard-data-run "<run_name>"] \
  [--light-model-dir "light_model_best"] \
  [--hard2-model-dir "hard_model_2nd_best"] \
  [--output-dir "14. best_model_result_and_mlp_final"] \
  [--run-name "<name>"] [--seed 42] \
  [--mlp-hidden-layers "32,16"] [--mlp-alpha 1e-4] [--mlp-max-iter 1200] [--val-size 0.3] \
  [--weight-light 0.55] [--weight-hard 0.45] [--weight-x 0.34] [--weight-y 0.33] [--weight-z 0.33] \
  [--target-suspect-recall 0.90] [--target-break-precision 0.90] \
  [--device auto|cpu|gpu]
```

### `15. make_save_model.py`
- Purpose: export deployable package for light/hard2/mlp with test artifacts
- Run:
```bash
python "make_ai/15. make_save_model.py" \
  [--light-dir "light_model_best"] \
  [--hard2-dir "hard_model_2nd_best"] \
  [--mlp-run "<14_run_name>"] \
  [--mlp-base-dir "14. best_model_result_and_mlp_final"] \
  [--output-dir "15. make_save_model"] \
  [--package-name "<name>"] \
  [--device auto|cpu|gpu]
```

### `16. test_ai.py`
- Purpose: production-style full pipeline from DB fetch to Excel prediction
- Output: Excel with `by_measure`, `by_pole` sheets
- Run:
```bash
python "make_ai/16. test_ai.py" \
  --server main|is|kh \
  [--project "<project_name>"] [--pole-id "<pole_id>"] \
  [--max-poles 0] [--min-points 200] [--max-points 400] \
  [--saved-package "<15_package_name>"] [--saved-base-dir "15. make_save_model"] \
  [--output-dir "16. test_ai"] \
  [--device auto|cpu|gpu]
```

## 6) Utilities

### `check_text_encoding.py`
- Purpose: check text encoding problems in files
- Run:
```bash
python "make_ai/check_text_encoding.py"
```

### `export_hard_2nd_conf_to_savedmodel.py`
- Purpose: export hard2 `.keras` confidence models to SavedModel format
- Run:
```bash
python "make_ai/export_hard_2nd_conf_to_savedmodel.py" \
  [--source-dir "<hard2_run_or_best_dir>"] \
  [--output-dir "<output_dir>"] \
  [--local]
```

### `export_hard_2nd_conf_to_savedmodel_wsl2.sh`
- Purpose: WSL2 wrapper for hard2 SavedModel export
- Run:
```bash
bash "make_ai/export_hard_2nd_conf_to_savedmodel_wsl2.sh"
```

### `hard_model_2nd_savedmodel_export.md`
- Purpose: detailed notes for hard2 SavedModel export

### `plot_processed_csv_2d.py`
- Purpose: plot one processed CSV as 2D contour/image
- Usage:
```python
from make_ai.plot_processed_csv_2d import plot_csv_2d
plot_csv_2d("sample.csv")
```

## Recommended Execution Order

1. `1` -> `2` -> `3` -> `3.1`
2. `4` -> `4.1` -> `5` -> `6`
3. `7` -> `8` (repeat until target performance)
4. `9` -> `10` -> `11`
5. `12` -> `13`
6. `14` -> `15` -> `16`

