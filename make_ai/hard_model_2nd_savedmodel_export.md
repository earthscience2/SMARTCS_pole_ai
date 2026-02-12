# Hard 2nd SavedModel Export

## 목적
`hard_model_2nd_best`(또는 특정 `12. hard_models_2nd/<run>`)의
`conf_x.keras`, `conf_y.keras`, `conf_z.keras`를
SavedModel 디렉터리로 변환한다.

출력 폴더 예시:
- `conf_x_savedmodel/`
- `conf_y_savedmodel/`
- `conf_z_savedmodel/`

## 기본 실행
```bash
python make_ai/export_hard_2nd_conf_to_savedmodel.py
```

## 특정 run에서 변환
```bash
python make_ai/export_hard_2nd_conf_to_savedmodel.py --source-dir "12. hard_models_2nd/20260205_2031"
```

## 출력 위치 지정
```bash
python make_ai/export_hard_2nd_conf_to_savedmodel.py \
  --source-dir "12. hard_models_2nd/20260205_2031" \
  --output-dir "15. make_save_model/tmp_export"
```

## Windows + WSL2 사용 시
Windows에서 실행하면 기본적으로 WSL2 실행 스크립트로 위임한다.
현재 환경에서 강제로 실행하려면 `--local`을 사용한다.

```bash
python make_ai/export_hard_2nd_conf_to_savedmodel.py --local
```
