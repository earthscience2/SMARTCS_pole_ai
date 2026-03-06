# main/ 실행 가이드

이 문서는 `main/` 하위 파이프라인의 최신 실행 순서를 간단히 정리합니다.  
세부 옵션은 각 폴더의 `README.md`를 우선 기준으로 봐주세요.

## 1) 데이터 준비

```bash
python "main/1. make_data set/1. get_project_info_list.py"
python "main/1. make_data set/2. get_anal_pole_list.py"
python "main/1. make_data set/3. get_raw_pole_data.py" --normal-ratio 10
python "main/1. make_data set/4. merge_data.py" --normal-ratio 10
python "main/1. make_data set/5. edit_data.py"
```

참고: `main/1. make_data set/README.md`

## 2) Light 모델

```bash
python "main/2. make_light_model/1. set_light_train_data.py"
python "main/2. make_light_model/2. make_light_model.py" --local
```

GPU 실행:

```bash
bash "main/2. make_light_model/2. make_light_model_gpu.sh"
```

참고: `main/2. make_light_model/README.md`

## 3) Hard 모델

```bash
python "main/3. make_hard_model/1. set_hard_train_data.py"
python "main/3. make_hard_model/2. make_hard_model_1st.py" --local
python "main/3. make_hard_model/3. make_hard_model_2nd.py" --local
```

GPU 실행:

```bash
bash "main/3. make_hard_model/2. make_hard_model_1st_gpu.sh"
bash "main/3. make_hard_model/3. make_hard_model_2nd_gpu.sh"
```

참고: `main/3. make_hard_model/README.md`

## 4) MLP 모델

```bash
python "main/4. make_mlp_model/1. mlp_train_data.py"
python "main/4. make_mlp_model/2. mlp_model.py" --local
```

GPU 실행:

```bash
bash "main/4. make_mlp_model/2. mlp_model_gpu.sh"
```

참고: `main/4. make_mlp_model/README.md`

## 5) 공통 원칙

- 학습 서브 파라미터는 CLI보다 각 스크립트의 `USER_OPTIONS`에서 관리합니다.
- 주요 로그는 `[정보]`, `[경고]`, `[오류]` 접두어를 사용합니다.
- 베스트 모델은 각 단계의 `best_*` 디렉터리와 히스토리 파일로 관리합니다.
