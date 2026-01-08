# pole_anal_system 폴더 파일 분석 결과

## 📋 파일 목록 및 필요성 분석

### ✅ 필수 파일 (핵심 기능) - 6개

| 파일명 | 필요성 | 사용 위치 | 주요 기능 |
|--------|--------|-----------|-----------|
| `pole_anal.py` | **필수** | `poleapp.py`에서 import | 전주 분석 핵심 로직, IN/OUT 데이터 분석, 3D/벡터 데이터 생성 |
| `pole_auto_analy_command.py` | **필수** | 모든 분석 스크립트에서 import | 전주 상태 스캔/분석, 신호 처리 함수, 핵심 분석 모듈 |
| `pole_analy_main.py` | **필수** | 직접 실행 | 명령행 인자 기반 메인 분석 실행 스크립트 |
| `pole_analy_main_schedule.py` | **필수** | `Pole_project_schedule_exe.py`에서 subprocess 호출 | 날짜 범위 지정 스케줄 기반 분석 실행 |
| `poleapp.py` | **필수** | 직접 실행 (웹 서버) | Flask 기반 REST API 서버, 분석 결과 제공 |
| `Pole_project_schedule_exe.py` | **필수** | 직접 실행 (스케줄러) | CSV 기반 프로젝트 자동 스케줄 실행 |

### ⚠️ 선택적 파일 (GUI 도구) - 4개

| 파일명 | 필요성 | 사용 위치 | 주요 기능 |
|--------|--------|-----------|-----------|
| `pole_auto_analy.py` | **선택** | GUI 직접 실행 | PyQt5 기반 메인 GUI 애플리케이션, 데이터 시각화 및 분석 |
| `pole_auto_analy_check.py` | **선택** | GUI 직접 실행 | 분석 전 검사용 GUI 도구, 데이터 검증 |
| `pole_auto_analy_ver2.py` | **선택** | GUI 직접 실행 | GUI v2 버전 (개선된 버전) |
| `pole_pre_scan_check.py` | **선택** | GUI 직접 실행 | 스캔 전 검사용 GUI 도구, 사전 점검 |

## 📊 상세 분석

### 1. `pole_anal.py` ✅ 필수
- **용도**: 전주 분석 핵심 로직 모듈
- **주요 함수**:
  - `get_pole_anal_result_in()`: 내부 측정 데이터 분석
  - `get_pole_anal_result_out()`: 외부 측정 데이터 분석
  - `get_pole_3d_data()`: 3D 차트 데이터 생성
  - `get_pole_vector_data()`: 벡터 데이터 생성
  - `pole_simulation()`: 자기장 시뮬레이션
- **사용처**: 
  - `poleapp.py`에서 import하여 웹 API 엔드포인트에서 사용
- **의존성**: `poledb`, `logger`, `magpylib`, `scipy`, `pandas`, `numpy`
- **결론**: **반드시 필요** (웹 API의 핵심 모듈)

### 2. `pole_auto_analy_command.py` ✅ 필수
- **용도**: 분석 명령어 및 신호 처리 모듈
- **주요 함수**:
  - `pole_State_Scan()`: 전주 상태 스캔
  - `pole_State_Analysis_all()`: 전주 상태 전체 분석
  - `conf_file_open()`: 설정 파일 열기
  - 신호 처리 함수들 (필터링, 피크 검출 등)
- **사용처**: 
  - `pole_analy_main.py`
  - `pole_analy_main_schedule.py`
  - `anal_data.py` (프로젝트 루트)
  - `SMARTCS_pole.py` (프로젝트 루트)
  - `simple_pole_anal.py` (프로젝트 루트)
  - 모든 GUI 파일들 (`pole_auto_analy*.py`, `pole_pre_scan_check.py`)
- **의존성**: `poledb`, `logger`, `slack`, `maintime`, `magpylib_simulation_m`
- **결론**: **반드시 필요** (가장 중요한 핵심 모듈, 모든 분석의 기반)

### 3. `pole_analy_main.py` ✅ 필수
- **용도**: 메인 분석 실행 스크립트 (명령행 인터페이스)
- **실행 방법**: 
  ```bash
  python pole_analy_main.py [프로젝트명] [방법] [옵션]
  ```
- **매개변수**:
  - `프로젝트명`: 분석할 프로젝트 그룹명
  - `방법`: `scan`, `analysis`, `all`
  - `옵션`: `all`, `MF`, `AP`, `AF`
- **출력**: `analysis/[프로젝트명]/output/` 디렉토리에 CSV 파일 생성
- **의존성**: `pole_auto_analy_command`
- **결론**: **반드시 필요** (CLI 사용 시 필수)

### 4. `pole_analy_main_schedule.py` ✅ 필수
- **용도**: 날짜 범위 지정 스케줄 기반 분석 실행
- **실행 방법**: 
  ```bash
  python pole_analy_main_schedule.py [프로젝트명] [방법] [옵션] [시작일] [종료일]
  ```
- **특징**: 
  - 날짜 범위를 지정하여 일별로 분석 수행
  - `start_date='now'` 옵션으로 현재 기준 역산 가능
  - 결과를 날짜별로 분리하여 저장
- **사용처**: `Pole_project_schedule_exe.py`에서 subprocess로 호출
- **출력**: `analysis/[프로젝트명]/output/schedule/` 디렉토리
- **의존성**: `pole_auto_analy_command`
- **결론**: **반드시 필요** (스케줄 기반 자동화 시 필수)

### 5. `poleapp.py` ✅ 필수
- **용도**: Flask 기반 REST API 웹 서버
- **주요 엔드포인트**:
  - `GET /polediag/getPoleDataAnalResult`: 전주 분석 결과 조회
  - 기타 분석 관련 API 엔드포인트
- **실행 방법**: 
  ```bash
  python poleapp.py
  ```
- **의존성**: `pole_anal`, `poledb`, `flask`
- **결론**: **반드시 필요** (웹 API 사용 시 필수)

### 6. `Pole_project_schedule_exe.py` ✅ 필수
- **용도**: 프로젝트 자동 스케줄 실행 스케줄러
- **실행 방법**: 직접 실행하여 백그라운드에서 실행
- **기능**: 
  - CSV 파일(`analysis/Pole_project_list.csv`)에서 프로젝트 목록 읽기
  - `state='ing'`인 프로젝트만 자동 실행
  - 매일 지정된 시간에 `pole_analy_main_schedule.py` 호출
- **스케줄 설정**: `schedule.every().day.at("21:43")` (코드 내 하드코딩)
- **의존성**: `schedule`, `subprocess`, `pandas`
- **결론**: **반드시 필요** (자동화 사용 시 필수)

### 7. `pole_auto_analy.py` ⚠️ 선택
- **용도**: PyQt5 기반 메인 GUI 애플리케이션
- **실행 방법**: 
  ```bash
  python pole_auto_analy.py
  ```
- **기능**: 
  - 전주 데이터 시각화
  - 그래프 표시 (pyqtgraph 사용)
  - 분석 실행 및 결과 확인
- **의존성**: `PyQt5`, `pyqtgraph`, `pole_auto_analy_command`
- **결론**: **GUI 사용 시 필요**, CLI만 사용 시 불필요

### 8. `pole_auto_analy_check.py` ⚠️ 선택
- **용도**: 분석 전 검사용 GUI 도구
- **실행 방법**: GUI 직접 실행
- **기능**: 
  - 분석 전 데이터 검증
  - 파일 업로드 기능
  - 분석 결과 확인
- **의존성**: `PyQt5`, `pyqtgraph`, `pole_auto_analy_command`
- **결론**: **GUI 사용 시 필요**, CLI만 사용 시 불필요

### 9. `pole_auto_analy_ver2.py` ⚠️ 선택
- **용도**: GUI v2 버전 (개선된 버전)
- **실행 방법**: GUI 직접 실행
- **특징**: 
  - v1의 개선 버전
  - v1과 기능 중복 가능
- **의존성**: `PyQt5`, `pyqtgraph`, `pole_auto_analy_command`
- **결론**: **v2 사용 시 필요**, v1 사용 시 중복 가능 (둘 중 하나만 유지 권장)

### 10. `pole_pre_scan_check.py` ⚠️ 선택
- **용도**: 스캔 전 검사용 GUI 도구
- **실행 방법**: GUI 직접 실행
- **기능**: 
  - 스캔 전 데이터 사전 점검
  - 데이터 유효성 검증
- **의존성**: `PyQt5`, `pyqtgraph`, `pole_auto_analy_command`
- **결론**: **GUI 사용 시 필요**, CLI만 사용 시 불필요

## 🔄 파일 간 의존성 관계

```
프로젝트 루트 스크립트들
    ↓
pole_auto_analy_command.py (핵심 모듈)
    ↓
    ├── pole_anal.py (분석 로직)
    │   └── poledb.py
    │
    ├── pole_analy_main.py (CLI 실행)
    │
    ├── pole_analy_main_schedule.py (스케줄 실행)
    │   └── Pole_project_schedule_exe.py (스케줄러)
    │
    └── GUI 파일들 (선택)
        ├── pole_auto_analy.py
        ├── pole_auto_analy_check.py
        ├── pole_auto_analy_ver2.py
        └── pole_pre_scan_check.py

poleapp.py (웹 서버)
    ↓
pole_anal.py
    ↓
poledb.py
```

## 🎯 권장 사항

### 최소 구성 (CLI만 사용)
```
pole_anal_system/
├── pole_anal.py                    ✅ 필수
├── pole_auto_analy_command.py      ✅ 필수
├── pole_analy_main.py              ✅ 필수
├── pole_analy_main_schedule.py    ✅ 필수
├── poleapp.py                      ✅ 필수 (웹 API 사용 시)
└── Pole_project_schedule_exe.py    ✅ 필수 (자동화 사용 시)
```

**총 6개 파일** (웹 API 미사용 시 5개, 자동화 미사용 시 4개)

### 전체 구성 (GUI 포함)
```
pole_anal_system/
├── pole_anal.py                    ✅ 필수
├── pole_auto_analy_command.py      ✅ 필수
├── pole_analy_main.py              ✅ 필수
├── pole_analy_main_schedule.py    ✅ 필수
├── poleapp.py                      ✅ 필수
├── Pole_project_schedule_exe.py   ✅ 필수
├── pole_auto_analy.py             ⚠️ 선택 (GUI)
├── pole_auto_analy_check.py       ⚠️ 선택 (GUI)
├── pole_auto_analy_ver2.py        ⚠️ 선택 (GUI v2) - v1과 중복
└── pole_pre_scan_check.py           ⚠️ 선택 (GUI)
```

**총 10개 파일** (GUI 사용 시)

### GUI 버전 선택 권장
- `pole_auto_analy.py` (v1) 또는 `pole_auto_analy_ver2.py` (v2) 중 **하나만 유지** 권장
- 둘 다 유지할 필요 없음

## 📝 실행 시나리오별 필요 파일

### 시나리오 1: CLI만 사용 (최소 구성)
**필요 파일**: 4개
- `pole_anal.py`
- `pole_auto_analy_command.py`
- `pole_analy_main.py`
- `pole_analy_main_schedule.py`

### 시나리오 2: CLI + 웹 API
**필요 파일**: 5개
- 시나리오 1 + `poleapp.py`

### 시나리오 3: CLI + 자동화
**필요 파일**: 5개
- 시나리오 1 + `Pole_project_schedule_exe.py`

### 시나리오 4: CLI + 웹 API + 자동화
**필요 파일**: 6개
- 시나리오 1 + `poleapp.py` + `Pole_project_schedule_exe.py`

### 시나리오 5: GUI 포함 (전체 구성)
**필요 파일**: 6~10개
- 시나리오 4 + GUI 파일들 (4개, 선택적)

## ✅ 결론

**필수 파일 (6개)**: 
- `pole_anal.py` - 분석 핵심 로직
- `pole_auto_analy_command.py` - 핵심 모듈 (가장 중요)
- `pole_analy_main.py` - CLI 실행
- `pole_analy_main_schedule.py` - 스케줄 실행
- `poleapp.py` - 웹 API (웹 사용 시)
- `Pole_project_schedule_exe.py` - 자동화 (자동화 사용 시)

**선택 파일 (4개)**: 
- GUI 관련 파일들 (GUI 사용 시에만 필요)
  - `pole_auto_analy.py` 또는 `pole_auto_analy_ver2.py` (둘 중 하나만 권장)

**최소 구성**: 4개 파일 (CLI만 사용)
**권장 구성**: 6개 파일 (CLI + 웹 API + 자동화)
**전체 구성**: 10개 파일 (GUI 포함)
