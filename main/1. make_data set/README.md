# 1. make_data set

전주 파단 예측용 학습 데이터셋 전처리 파이프라인입니다.  
실행 흐름은 `DB 조회 -> 원본 수집 -> 정규화/병합 -> ROI 수동 보정` 순서입니다.

## 전체 흐름

1. `1. get_project_info_list.py`  
프로젝트 목록과 분석 진행 통계를 서버별로 수집
2. `2. get_anal_pole_list.py`  
분석 완료 전주 목록(파단/정상 라벨 포함) 수집
3. `3. get_raw_pole_data.py`  
원본 측정 CSV(IN/OUT)와 전주별 정보 JSON 저장
4. `4. merge_data.py`  
원본 CSV를 학습용 격자 데이터(`*_OUT_processed.csv`)로 변환
5. `5. edit_data.py`  
GUI에서 ROI(관심영역) 수동 보정 및 삭제 처리

보조 점검 스크립트:
- `3.1. check_raw_pole_data_info.py`
- `4.1. check_merge_data_info.py`

## 디렉터리 구조(핵심)

- `1. project_info_list/`: 1단계 결과 JSON
- `2. anal_pole_list/`: 2단계 결과 JSON
- `3. raw_pole_data/`: 3단계 원본 데이터
- `4. merge_data/`: 4단계 전처리 데이터
- `5. edit_data/`: 5단계 ROI 보정 결과

## 실행 전 준비

- Python 3.10+ 권장
- 필수 패키지: `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm`
- GUI 편집용: `tkinter` 사용 가능 환경
- DB 접속 설정 필요:
  - 프로젝트 루트의 `config/poledb.py`
  - `main/is/kh` 서버에 대한 DB 접속 가능 상태

## 단계별 사용법

아래 명령은 `main/1. make_data set` 폴더에서 실행 기준입니다.

### 1) 프로젝트 통계 수집

```bash
python "1. get_project_info_list.py" --server all
```

주요 옵션:
- `--server {all,main,is,kh}`: 조회 서버
- `--output-dir`: 결과 저장 폴더(기본 `1. project_info_list`)

출력:
- `project_list_all_YYYYMMDD_HHMM.json`

### 2) 분석 완료 전주 목록 수집

```bash
python "2. get_anal_pole_list.py" --server all
```

주요 옵션:
- `--project-list-json`: 1단계 결과 JSON 직접 지정
- `--project-list-dir`: 1단계 JSON 탐색 폴더
- `--output-dir`: 결과 저장 폴더(기본 `2. anal_pole_list`)

출력:
- `anal2_poles_all_YYYYMMDD_HHMM.json`

### 3) 원본 측정 데이터 수집

```bash
python "3. get_raw_pole_data.py" --normal-ratio 10
```

주요 옵션:
- `--input-json`: 2단계 결과 JSON 지정
- `--input-dir`: 2단계 JSON 탐색 폴더
- `--output-dir`: 결과 저장 폴더(기본 `3. raw_pole_data`)
- `--normal-ratio`: 정상:파단 수집 비율(기본 10배)

출력:
- `3. raw_pole_data/break/...`
- `3. raw_pole_data/normal/...`
- `raw_pole_data_summary_YYYYMMDD_HHMM.json`

### 3-1) 원본 수집 상태 점검

```bash
python "3.1. check_raw_pole_data_info.py"
python "3.1. check_raw_pole_data_info.py" --no-plot
```

### 4) 학습용 merge 데이터 생성

```bash
python "4. merge_data.py" --normal-ratio 10
```

주요 옵션:
- `--raw-data-dir`: 입력 폴더(기본 `3. raw_pole_data`)
- `--output-dir`: 출력 폴더(기본 `4. merge_data`)
- `--normal-ratio`: 정상 샘플 비율

출력:
- `4. merge_data/break/<project>/<poleid>/*_OUT_processed.csv`
- `4. merge_data/break/.../*_OUT_processed_break_info.json`
- `4. merge_data/break/.../*_OUT_processed_2d_plot.png`(가능 시)
- `4. merge_data/normal/<project>/<poleid>/*_OUT_processed.csv`

### 4-1) merge 데이터 점검

```bash
python "4.1. check_merge_data_info.py"
python "4.1. check_merge_data_info.py" --no-plot
```

### 5) ROI 수동 보정 GUI

```bash
python "5. edit_data.py"
```

주요 옵션:
- `--input-dir`: 편집 대상 폴더(기본 `4. merge_data/break`)

GUI 핵심 동작:
- ROI 드래그로 설정
- `S`: 저장
- 화살표 키: 이전/다음 이미지
- 삭제 처리 가능(`deleted` 플래그)

결과 저장:
- `5. edit_data/break/<project>/<poleid>/*_roi_info.json`
- 필요 시 확정 파일(`csv`, `break_info`, `png`) 복사

## 추천 실행 순서

```bash
python "1. get_project_info_list.py"
python "2. get_anal_pole_list.py"
python "3. get_raw_pole_data.py" --normal-ratio 10
python "3.1. check_raw_pole_data_info.py" --no-plot
python "4. merge_data.py" --normal-ratio 10
python "4.1. check_merge_data_info.py" --no-plot
python "5. edit_data.py"
```

## 로그 형식

- 주요 실행 로그는 `[정보]`, `[경고]`, `[오류]` 접두어를 사용합니다.
- 단계별 시작/완료 구분선을 함께 출력합니다.

## 참고

- 파일명이 `1. xxx.py` 형태라서 실행 시 따옴표를 권장합니다.
- 2단계/3단계는 입력 파일을 지정하지 않으면 최신 JSON을 자동 선택합니다.
- 기존 결과가 있으면 일부 단계는 이미 저장된 데이터를 건너뜁니다(증분 수집).
