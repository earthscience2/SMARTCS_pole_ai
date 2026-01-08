# 전주분석 프로젝트

전주 데이터를 바탕으로 데이터 수집 및 분석을 수행하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 전주(전기주)의 진단 데이터를 수집하고 분석하여 전주의 상태를 평가하고 파단 여부를 판단하는 시스템입니다. MySQL 데이터베이스에 저장된 전주 측정 데이터를 기반으로 자동화된 분석을 수행합니다.

## 🚀 주요 기능

### 1. 데이터 수집 및 관리
- 전주 측정 데이터 수집
- 데이터베이스 연동 (MySQL)
- 원본 데이터 추출 및 변환

### 2. 전주 상태 분석
- **스캔 (Scan)**: 전주 신호 적합성 분석
- **분석 (Analysis)**: 전주 신호 파단 여부 분석
- 1차/2차 분석 지원

### 3. 데이터 처리
- CSV/Excel 형식으로 데이터 출력
- AI 학습용 데이터셋 생성
- 분석 결과 상세 리포트 생성

### 4. 자동화
- 스케줄 기반 자동 분석
- Slack 알림 연동
- 배치 처리 지원

## 📁 프로젝트 구조

> 각 파일의 상세 기능은 [CODE_SUMMARY.md](CODE_SUMMARY.md)를 참고하세요.

```
SMARTCS/
├── anal_data.py              # 자동 분석 데이터 처리
├── export_data.py            # 데이터 내보내기 및 정보 조회
├── make_ai_set.py            # AI 학습용 데이터셋 생성
├── make_data.py              # 원본 데이터 추출 (Excel)
├── make_data_csv.py          # 원본 데이터 추출 (CSV)
├── make_data_detail.py       # 상세 데이터 추출
├── make_data_set.py          # 데이터셋 생성
├── pole_anal.py              # 전주 분석 핵심 로직
├── pole_analy_main.py        # 메인 분석 실행 스크립트
├── pole_auto_analy.py        # 자동 분석 모듈
├── pole_auto_analy_command.py # 분석 명령어 모듈
├── poledb.py                 # 데이터베이스 연결 모듈
├── poleconf.py               # 설정 파일
├── mysqldb.py                # MySQL 핸들러
├── logger.py                 # 로깅 모듈
├── slack.py                  # Slack 알림 모듈
├── maintime.py               # 시간 처리 모듈
├── AUTO_analysis/            # 자동 분석 결과 저장 디렉토리
├── ai_data_set/              # AI 학습용 데이터셋
├── polelist/                 # 전주 목록 및 원본 데이터
├── log/                      # 로그 파일
└── requirements.txt          # 패키지 의존성
```

## 🔧 설치 및 설정

### 1. 필수 요구사항
- Python 3.7 이상
- MySQL 데이터베이스

### 2. 가상환경 생성 (권장)

프로젝트를 격리된 환경에서 실행하기 위해 가상환경을 생성하는 것을 권장합니다.

#### Windows
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 가상환경 비활성화 (필요시)
deactivate
```

#### Mac/Linux
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 가상환경 비활성화 (필요시)
deactivate
```

> **참고**: 가상환경이 활성화되면 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

### 3. 패키지 설치

#### 최소 필수 패키지 설치 (권장)
```bash
pip install -r requirements-minimal.txt
```

#### 전체 패키지 설치 (GUI, 웹서버 포함)
```bash
pip install -r requirements.txt
```

**주요 패키지:**
- `pandas`: 데이터 처리
- `numpy`: 수치 연산
- `scipy`: 과학 계산
- `PyMySQL`: MySQL 연결
- `openpyxl`: Excel 파일 처리
- `magpylib`: 자기장 시뮬레이션
- `bottleneck`: 성능 최적화
- `PeakUtils`: 피크 검출
- `requests`: HTTP 요청 (Slack 알림)

자세한 설치 가이드는 [INSTALL.md](INSTALL.md)를 참고하세요.

### 4. 데이터베이스 설정

`poleconf.py` 파일에서 데이터베이스 연결 정보를 설정합니다:

```python
poledb_host = 'your_host:port'
poledb_dbname = 'your_database'
poledb_user = 'your_username'
poledb_pwd = 'your_password'
```

또는 `pole.ini` 파일을 사용하여 설정할 수 있습니다.

## 📖 사용 방법

### 1. 전주 분석 실행

```bash
python pole_analy_main.py [프로젝트명] [방법] [옵션]
```

**매개변수:**
- `프로젝트명`: 분석할 프로젝트 그룹명 (예: `경기안산-202208`)
- `방법`: 
  - `scan`: 전주 신호 적합성 분석
  - `analysis`: 전주 신호 파단여부 분석
  - `all`: 스캔 + 분석 모두 수행
- `옵션`:
  - `all`: 모든 전주
  - `MF`: 측정완료 전주
  - `AP`: 1차 분석 완료 전주
  - `AF`: 2차 분석 완료 전주

**예시:**
```bash
python pole_analy_main.py 경기안산-202208 all all
```

### 2. 자동 분석 실행

```bash
python anal_data.py
```

### 3. 데이터 추출

#### Excel 형식으로 원본 데이터 추출
```python
python make_data.py
```

#### CSV 형식으로 원본 데이터 추출
```python
python make_data_csv.py
```

#### 상세 데이터 추출
```python
python make_data_detail.py
```

### 4. AI 데이터셋 생성

```python
python make_ai_set.py
```

## 📊 분석 상태 코드

### 측정 관련
- `-`: 미측정
- `MF`: 측정완료
- `AP`: 분석중 (1차분석 완료)
- `AF`: 분석완료

### 측정 결과
- `N`: 정상
- `B`: 파단
- `U`: 보류
- `X`: 측정불가

### 분석 관련
- `NA`: 미대상
- `-`: 미분석
- `AP`: 분석중
- `AF`: 분석완료

## 📂 출력 파일

분석 결과는 다음 경로에 저장됩니다:

- **자동 분석 결과**: `./AUTO_analysis/[날짜]_result/[프로젝트명]/output/`
- **AI 데이터셋**: `./ai_data_set/`
- **원본 데이터**: `./polelist/[프로젝트명] 원본데이터/`

## 🔍 주요 모듈 설명

### `pole_anal.py`
전주 분석의 핵심 로직을 포함합니다:
- `get_pole_anal_result_in()`: 내부 측정 데이터 분석
- `get_pole_anal_result_out()`: 외부 측정 데이터 분석
- `get_pole_3d_data()`: 3D 데이터 생성
- `get_pole_vector_data()`: 벡터 데이터 생성

### `export_data.py`
데이터베이스에서 정보를 조회하는 함수들을 제공합니다:
- `groupname_info()`: 그룹명 목록 조회
- `group_diag_progress_info()`: 측정 진행도 조회
- `group_anal_progress_info()`: 분석 진행도 및 결과 조회

### `poledb.py`
데이터베이스 연결 및 데이터 조회 함수를 제공합니다.

## 🛠️ 개발 환경

- Python 3.7+
- MySQL 5.7+
- 주요 라이브러리: pandas, numpy, scipy, PyMySQL

## 📝 로그

로그 파일은 `log/` 디렉토리에 저장되며, 날짜별로 관리됩니다.

## ⚠️ 주의사항

1. 데이터베이스 연결 정보는 보안을 위해 환경 변수나 설정 파일로 관리하는 것을 권장합니다.
2. 대용량 데이터 처리 시 메모리 사용량에 주의하세요.
3. 분석 결과는 정기적으로 백업하세요.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

## 📄 라이선스

이 프로젝트는 내부 사용을 위한 프로젝트입니다.

