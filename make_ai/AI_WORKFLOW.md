# AI 데이터 수집 및 학습 워크플로우

## 📊 전체 워크플로우 개요

```
1. 원본 데이터 수집
   ↓
2. 정상/파단 전주 목록 수집 (JSON)
   ↓
3. AI 학습용 데이터 변환 (CSV)
   ↓
4. 데이터 검증 (시각화 및 수동 확인)
   ↓
5. AI 모델 학습
   ↓
6. 학습된 모델로 예측 수행
```

---

## 🔍 1단계: AI 데이터 수집 부분

### 1-1. 원본 데이터 수집

#### `smart_get_raw_data.py` - 전체 프로젝트 원본 데이터 수집
- **기능**: 프로젝트 전체의 전주 원본 측정 데이터를 수집
- **입력**: 프로젝트 그룹명 리스트
- **출력**: 
  - `raw_data/[프로젝트명] 원본데이터/[프로젝트명] csv/` - CSV 파일
  - `raw_data/[프로젝트명] 원본데이터/[프로젝트명] xlsx/` - Excel 파일
- **데이터 형식**: 
  - IN 데이터: `{프로젝트명}_{전주ID}_{측정번호}_{날짜}_T.csv`
  - OUT 데이터: `{프로젝트명}_{전주ID}_{측정번호}_{날짜}_H.csv`
- **실행 방법**:
  ```python
  # 코드 내 name_list 수정 후 실행
  name_list = ["천안지사2-2511", "아산지사2-2511"]
  server = "is"  # main, is, kh 중 선택
  ```

#### `smart_get_raw_data_poleid.py` - 특정 전주 ID 원본 데이터 수집
- **기능**: 특정 전주 ID의 원본 데이터만 수집
- **입력**: 프로젝트 그룹명, 전주 ID 리스트
- **출력**: `raw_data/[프로젝트명] [전주ID] 원본데이터/`
- **실행 방법**:
  ```python
  name_list = "노원도봉-2503"
  poleid_name_list = ["0432R025", "0429P091"]
  ```

---

### 1-2. 정상/파단 전주 목록 수집

#### `smart_get_ai_data_json.py` - 정상/파단 전주 목록 JSON 생성
- **기능**: 데이터베이스에서 정상(N)과 파단(B) 전주 목록을 JSON으로 수집
- **입력**: 서버 리스트 (main, is, kh)
- **출력**: 
  - `raw_data_ai_json/normal/[그룹명].json` - 정상 전주 목록
  - `raw_data_ai_json/break/[그룹명].json` - 파단 전주 목록 (전주ID, 파단높이, 파단각도)
- **데이터 구조**:
  - 정상: `["전주ID1", "전주ID2", ...]`
  - 파단: `[["전주ID1", "높이", "각도"], ...]`
- **실행 방법**: 직접 실행

---

### 1-3. AI 학습용 데이터 변환

#### `smart_get_ai_data_csv.py` - JSON 기반 CSV 데이터 생성
- **기능**: JSON 파일을 기반으로 정상/파단 데이터를 CSV로 변환
- **입력**: 
  - `raw_data_ai_json/normal/`, `raw_data_ai_json/break/` 폴더의 JSON 파일
  - 데이터베이스의 원본 측정 데이터
- **출력**: 
  - `raw_data_ai_csv/normal/[그룹명]/` - 정상 데이터 CSV
  - `raw_data_ai_csv/break/[그룹명]/` - 파단 데이터 CSV
- **파일 형식**: 
  - `{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_x.csv`
  - `{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_y.csv`
  - `{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_z.csv`
- **특징**: 
  - 이미 존재하는 파일은 건너뛰기
  - 파단 위치가 측정 범위 내에 있는 경우만 파단 데이터로 분류
- **실행 방법**: 직접 실행

---

### 1-4. 데이터 검증 (수동 확인)

#### `smart_check_break.py` - 파단 데이터 검증
- **기능**: 파단 데이터를 시각화하여 수동으로 검증
- **입력**: `raw_data_ai_csv/break/` 폴더의 CSV 파일
- **출력**: 
  - `raw_data_ai_csv_check/break/` - 검증된 파단 데이터 (y 입력 시)
  - `raw_data_ai_csv_check/other/` - 비파단 데이터 (n 입력 시)
- **프로세스**:
  1. 파단 위치 기준 ±10% 범위의 데이터 추출
  2. X, Y, Z 축 데이터를 그래프로 시각화
  3. 사용자가 'y' 또는 'n' 키 입력으로 검증
  4. 검증 결과에 따라 분류하여 저장
- **실행 방법**: 직접 실행 (인터랙티브)

#### `smart_check_normal.py` - 정상 데이터 검증
- **기능**: 정상 데이터를 시각화하여 수동으로 검증
- **입력**: `raw_data_ai_csv/normal/` 폴더의 CSV 파일
- **출력**: 
  - `raw_data_ai_csv_check/normal/` - 검증된 정상 데이터 (y 입력 시)
  - `raw_data_ai_csv_check/other/` - 비정상 데이터 (n 입력 시)
- **프로세스**: `smart_check_break.py`와 동일
- **실행 방법**: 직접 실행 (인터랙티브)

#### `smart_recheck_break.py` - 파단 데이터 재검증
- **기능**: 파단 데이터를 다시 검증
- **입력/출력**: `smart_check_break.py`와 동일
- **실행 방법**: 직접 실행 (인터랙티브)

---

## 🤖 2단계: AI 학습 부분

### 2-1. AI 모델 학습

#### `smart_ai_study.py` - AI 모델 학습
- **기능**: 검증된 데이터로 딥러닝 모델 학습
- **입력**: 
  - `raw_data_ai_csv_check/normal/` - 정상 데이터 (label=0)
  - `raw_data_ai_csv_check/break/` - 파단 데이터 (label=1)
- **모델 구조**:
  - **아키텍처**: Conv1D + Bidirectional LSTM + Dense
  - **레이어 구성**:
    1. Conv1D (128 filters, kernel_size=5)
    2. BatchNormalization
    3. MaxPooling1D
    4. Bidirectional LSTM (128 units, return_sequences=True)
    5. Bidirectional LSTM (64 units)
    6. Dropout (0.5)
    7. Dense (64 units)
    8. Dense (32 units)
    9. Dense (1 unit, sigmoid) - 이진 분류
  - **시퀀스 길이**: 30
  - **배치 크기**: 32
  - **에폭**: 50 (EarlyStopping 적용)
- **전처리**:
  - StandardScaler로 피처 스케일링
  - SMOTE로 데이터 불균형 해소
  - 시퀀스 생성 (길이 30)
- **학습 설정**:
  - Optimizer: Adam (Gradient Clipping 적용)
  - Loss: binary_crossentropy
  - Learning Rate Scheduler: 동적 조정
  - EarlyStopping: val_loss 모니터링, patience=10
- **평가**:
  - ROC Curve, AUC 계산
  - Precision-Recall Curve로 최적 임계값 찾기
  - Classification Report 출력
- **출력**: 
  - `result_ai/model_67.keras` - 학습된 모델 파일
- **실행 방법**: 직접 실행

---

### 2-2. 학습된 모델로 예측

#### `smart_anal_ai.py` - AI 모델 예측 수행
- **기능**: 학습된 모델을 사용하여 전주 데이터의 파단 확률 예측
- **입력**: 
  - 학습된 모델 파일 (`result_ai/model_38.keras`)
  - 데이터베이스의 전주 측정 데이터
- **프로세스**:
  1. 지정된 그룹의 모든 전주 데이터 로드
  2. 각 전주의 OUT 측정 데이터 (x, y, z) 추출
  3. 데이터를 시퀀스 단위로 분할 (길이 30)
  4. 모델로 예측 수행
  5. 각 전주의 최대 파단 확률 계산
  6. 파단 확률 기준으로 정렬
- **출력**: 
  - 콘솔에 전주 ID, 측정 번호, 파단 확률 출력
  - 파단 확률이 높은 순서대로 정렬된 리스트
- **실행 방법**:
  ```python
  # 코드 내 설정 수정 후 실행
  server = "main"
  group_name = "강원강릉-202306"
  model_name = "model_38"
  ```

---

## 📋 실행 순서

### 초기 데이터 수집
1. `smart_get_raw_data.py` - 원본 데이터 수집
2. `smart_get_ai_data_json.py` - 정상/파단 전주 목록 수집
3. `smart_get_ai_data_csv.py` - AI 학습용 CSV 데이터 생성

### 데이터 검증
4. `smart_check_break.py` - 파단 데이터 검증
5. `smart_check_normal.py` - 정상 데이터 검증
6. (선택) `smart_recheck_break.py` - 파단 데이터 재검증

### AI 학습 및 예측
7. `smart_ai_study.py` - AI 모델 학습
8. `smart_anal_ai.py` - 학습된 모델로 예측 수행

---

## 📁 디렉토리 구조

```
make_ai/
├── raw_data/                          # 원본 데이터
│   └── [프로젝트명] 원본데이터/
│       ├── [프로젝트명] csv/
│       └── [프로젝트명] xlsx/
│
├── raw_data_ai_json/                  # 정상/파단 전주 목록
│   ├── normal/
│   │   └── [그룹명].json
│   └── break/
│       └── [그룹명].json
│
├── raw_data_ai_csv/                   # AI 학습용 원본 CSV
│   ├── normal/
│   │   └── [그룹명]/
│   └── break/
│       └── [그룹명]/
│
├── raw_data_ai_csv_check/             # 검증된 데이터
│   ├── normal/
│   ├── break/
│   └── other/
│
└── result_ai/                         # 학습 결과
    └── model_*.keras
```

---

## 🔑 핵심 포인트

### 데이터 수집 부분
- **원본 데이터**: 데이터베이스에서 전주 측정 데이터 추출
- **레이블링**: 데이터베이스의 분석 결과(N/B)를 기반으로 정상/파단 분류
- **검증**: 시각화를 통한 수동 검증으로 데이터 품질 보장

### AI 학습 부분
- **모델**: 시계열 데이터를 위한 Conv1D + LSTM 하이브리드 모델
- **전처리**: 스케일링, 시퀀스 생성, 데이터 증강(SMOTE)
- **예측**: 전주별 파단 확률 계산 및 정렬

---

## ⚙️ 설정 변경 사항

각 파일 실행 전 다음 설정을 확인/수정해야 합니다:

1. **서버 설정**: `server = "main"` 또는 `"is"`, `"kh"`
2. **프로젝트/그룹명**: `name_list`, `group_name` 등
3. **모델명**: `model_name` (학습 시 자동 생성, 예측 시 지정)
4. **경로**: 상대 경로가 프로젝트 루트 기준인지 확인

