# 프로젝트 구조 및 데이터 저장 구조

## 📋 프로젝트 개요

전주분석 프로젝트는 여러 서버에 분산된 전주 진단 데이터를 수집, 분석, 관리하는 시스템입니다.

## 🖥️ 서버 정보

| 서버 코드 | 서버 이름 | 호스트 주소 | 설명 |
|----------|----------|------------|------|
| `main` | 메인서버 | 210.105.85.3 | 메인 데이터베이스 서버 |
| `is` | 이수서버 | smartpole-is.iptime.org:33306 | 이수 서버 |
| `kh` | 건화서버 | smartpole-kh.iptime.org:33306 | 건화 서버 |
| `jt` | 제이티엔지니어링 | smartpole-jt.iptime.org:33306 | 제이티엔지니어링 서버 |

**공통 설정:**
- 데이터베이스명: `polediagdb`
- 사용자명: `polediag`
- 포트: 기본 3306 (is, kh, jt는 33306)

## 📁 프로젝트 구조

### 프로젝트 명명 규칙
프로젝트 이름은 다음과 같은 형식을 따릅니다:
- 예시: `아산지사2-2511`, `천안지사2-2511`, `강원강릉-202306`
- 형식: `[지역명]-[날짜]` 또는 `[지역명][번호]-[날짜]`
- 날짜 형식: `YYMM` (예: 2511 = 2025년 11월)

### 디렉토리 구조

```
SMARTCS/
├── get_pole_date/                    # 데이터 수집 관련
│   ├── get_project_list.py          # 프로젝트 목록 조회 및 저장
│   └── get_raw_pole_data.py         # 원본 데이터 수집
│
├── make_ai/                          # AI 학습 관련
│   ├── raw_data/                     # 원본 데이터
│   │   └── [프로젝트명] 원본데이터/
│   │       ├── [프로젝트명] csv/
│   │       └── [프로젝트명] xlsx/
│   │
│   ├── raw_data_ai_json/            # 정상/파단 전주 목록
│   │   ├── normal/
│   │   │   └── [프로젝트명].json
│   │   └── break/
│   │       └── [프로젝트명].json
│   │
│   ├── raw_data_ai_csv/              # AI 학습용 원본 CSV
│   │   ├── normal/
│   │   │   └── [프로젝트명]/
│   │   └── break/
│   │       └── [프로젝트명]/
│   │
│   ├── raw_data_ai_csv_check/       # 검증된 데이터
│   │   ├── normal/
│   │   ├── break/
│   │   └── other/
│   │
│   └── result_ai/                    # 학습 결과
│       └── model_*.keras
│
├── project_list/                     # 프로젝트 목록 저장
│   ├── project_list_main_[timestamp].json
│   ├── project_list_is_[timestamp].json
│   ├── project_list_kh_[timestamp].json
│   ├── project_list_jt_[timestamp].json
│   └── project_list_all_[timestamp].json
│
├── AUTO_analysis/                    # 자동 분석 결과
│   └── [날짜]_result/
│       └── [프로젝트명]/
│           ├── input/
│           └── output/
│
└── polelist/                         # 전주 목록 및 원본 데이터
    └── [프로젝트명] 원본데이터/
```

## 🗄️ 데이터베이스 구조

### 주요 테이블

#### 1. `tb_pole_group`
프로젝트(그룹) 목록을 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `groupname` | VARCHAR | 프로젝트 이름 (예: "아산지사2-2511") |

**조회 방법:**
```python
PDB.groupname_info()  # 모든 프로젝트 목록 반환
```

#### 2. `tb_pole`
전주 기본 정보를 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `poleid` | VARCHAR | 전주 ID (고유 식별자) |
| `groupname` | VARCHAR | 프로젝트 이름 |
| `regdate` | DATETIME | 등록일시 |
| `diagstate` | VARCHAR | 진단 상태 (-, MF, AP, AF) |

**조회 방법:**
```python
PDB.get_pole_list(groupname)  # 특정 프로젝트의 전주 목록
```

#### 3. `tb_diag_state`
전주 진단 상태 정보를 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `poleid` | VARCHAR | 전주 ID |
| `groupname` | VARCHAR | 프로젝트 이름 |
| `breakstate` | VARCHAR | 파단 상태 (N, B, U, X) |
| `endtime` | DATETIME | 측정 종료 시간 |
| `teamid` | VARCHAR | 팀 ID |

**상태 코드:**
- `N`: 정상
- `B`: 파단
- `U`: 보류
- `X`: 측정불가
- `-`: 없음

#### 4. `tb_anal_result`
전주 분석 결과를 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `poleid` | VARCHAR | 전주 ID |
| `groupname` | VARCHAR | 프로젝트 이름 |
| `analstep` | INT | 분석 단계 (1: 1차 분석, 2: 2차 분석) |
| `breakstate` | VARCHAR | 파단 상태 (N, B, U, X) |
| `breakheight` | FLOAT | 파단 높이 (m) |
| `breakdegree` | FLOAT | 파단 각도 (도) |

**조회 방법:**
```python
PDB.group_anal_type_pole_2(groupname, "N")  # 정상 전주 목록
PDB.group_anal_type_pole_2(groupname, "B")  # 파단 전주 목록
```

#### 5. `tb_diag_pole_meas_result`
전주 측정 결과 정보를 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `poleid` | VARCHAR | 전주 ID |
| `measno` | INT | 측정 번호 |
| `devicetype` | VARCHAR | 장치 타입 (IN, OUT) |
| `stdegree` | FLOAT | 시작 각도 |
| `eddegree` | FLOAT | 종료 각도 |
| `stheight` | FLOAT | 시작 높이 (m) |
| `edheight` | FLOAT | 종료 높이 (m) |
| `sttime` | DATETIME | 측정 시작 시간 |
| `endtime` | DATETIME | 측정 종료 시간 |

**조회 방법:**
```python
PDB.get_meas_result(poleid, devicetype)  # 특정 전주의 측정 결과
```

#### 6. `tb_diag_pole_meas_data`
전주 측정 원시 데이터를 저장하는 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `poleid` | VARCHAR | 전주 ID |
| `measno` | INT | 측정 번호 |
| `devicetype` | VARCHAR | 장치 타입 (IN, OUT) |
| `axis` | VARCHAR | 축 (x, y, z) |
| `idx` | INT | 인덱스 |
| `ch1` ~ `ch8` | FLOAT | 채널 1~8의 측정값 |

**조회 방법:**
```python
PDB.get_meas_data(poleid, measno, devicetype, axis)  # 측정 데이터 조회
```

## 📊 파단 정보 저장 구조

### JSON 파일 구조

#### 정상 전주 목록 (`raw_data_ai_json/normal/[프로젝트명].json`)
```json
[
  "전주ID1",
  "전주ID2",
  "전주ID3"
]
```

#### 파단 전주 목록 (`raw_data_ai_json/break/[프로젝트명].json`)
```json
[
  ["전주ID1", "파단높이(m)", "파단각도(도)"],
  ["전주ID2", "1.5", "90.0"],
  ["전주ID3", "2.3", "180.0"]
]
```

**데이터 구조:**
- `[0]`: 전주 ID
- `[1]`: 파단 높이 (미터 단위, 문자열)
- `[2]`: 파단 각도 (도 단위, 문자열)

### CSV 파일 구조

#### 원본 데이터 파일명 형식
```
{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_x.csv
{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_y.csv
{전주ID}_{측정번호}_{날짜}_{시작각도}_{종료각도}_{시작높이}_{종료높이}_H_z.csv
```

**예시:**
```
0432R025_1_2025-11-15_0_90_0.0_1.5_H_x.csv
```

**파일명 구성 요소:**
- `전주ID`: 전주 고유 식별자
- `측정번호`: 측정 횟수 (1, 2, 3, ...)
- `날짜`: 측정 날짜 (YYYY-MM-DD)
- `시작각도`: 측정 시작 각도
- `종료각도`: 측정 종료 각도
- `시작높이`: 측정 시작 높이 (m)
- `종료높이`: 측정 종료 높이 (m)
- `H`: OUT 측정 데이터 (T는 IN 측정 데이터)
- `x/y/z`: 측정 축

#### CSV 파일 내용 구조

**컬럼 구조:**
```
idx, measno, poleid, groupname, devicetype, axis, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8
```

**데이터 예시:**
```csv
idx,measno,poleid,groupname,devicetype,axis,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
0,1,0432R025,아산지사2-2511,OUT,x,123.45,234.56,345.67,456.78,567.89,678.90,789.01,890.12
1,1,0432R025,아산지사2-2511,OUT,x,124.45,235.56,346.67,457.78,568.89,679.90,790.01,891.12
...
```

## 🔄 데이터 흐름

### 1. 프로젝트 목록 조회
```
서버 (main/is/kh/jt)
  ↓
tb_pole_group 테이블 조회
  ↓
JSON 파일 저장 (project_list/)
```

### 2. 전주 데이터 수집
```
프로젝트 선택
  ↓
tb_pole 테이블에서 전주 목록 조회
  ↓
각 전주의 tb_diag_pole_meas_data 조회
  ↓
CSV/Excel 파일로 저장
```

### 3. 파단 정보 수집
```
프로젝트 선택
  ↓
tb_anal_result 테이블에서 분석 결과 조회
  ↓
정상(N) / 파단(B) 전주 분류
  ↓
JSON 파일로 저장 (raw_data_ai_json/)
```

### 4. AI 학습용 데이터 생성
```
JSON 파일 (정상/파단 목록)
  ↓
tb_diag_pole_meas_data에서 실제 측정 데이터 조회
  ↓
CSV 파일로 저장 (raw_data_ai_csv/)
  ↓
수동 검증 (smart_check_*.py)
  ↓
검증된 데이터 저장 (raw_data_ai_csv_check/)
```

## 📝 프로젝트 목록 JSON 파일 구조

### 개별 서버 파일 (`project_list_[서버]_[timestamp].json`)
```json
{
  "server": "is",
  "server_name": "이수서버",
  "timestamp": "20250115_143022",
  "total_count": 150,
  "projects": [
    "아산지사2-2511",
    "천안지사2-2511",
    "강원강릉-202306",
    ...
  ]
}
```

### 전체 통합 파일 (`project_list_all_[timestamp].json`)
```json
{
  "timestamp": "20250115_143022",
  "servers": {
    "main": {
      "server_name": "메인서버",
      "projects": ["프로젝트1", "프로젝트2", ...],
      "count": 200
    },
    "is": {
      "server_name": "이수서버",
      "projects": ["프로젝트1", "프로젝트2", ...],
      "count": 150
    },
    "kh": {
      "server_name": "건화서버",
      "projects": ["프로젝트1", "프로젝트2", ...],
      "count": 100
    },
    "jt": {
      "server_name": "제이티엔지니어링",
      "projects": ["프로젝트1", "프로젝트2", ...],
      "count": 50
    }
  },
  "summary": {
    "main": 200,
    "is": 150,
    "kh": 100,
    "jt": 50
  },
  "total_projects": 500
}
```

## 🔍 주요 함수 및 사용법

### 프로젝트 목록 조회
```python
import poledb as PDB

# 서버 초기화
PDB.poledb_init("is")  # main, is, kh, jt 중 선택

# 프로젝트 목록 조회
project_list = PDB.groupname_info()
# 반환: ["아산지사2-2511", "천안지사2-2511", ...]
```

### 전주 목록 조회
```python
# 특정 프로젝트의 전주 목록 조회
pole_list = PDB.get_pole_list("아산지사2-2511")
# 반환: DataFrame (poleid, regdate, diagstate, endtime, breakstate, teamid)
```

### 파단 정보 조회
```python
# 정상 전주 목록
normal_poles = PDB.group_anal_type_pole_2("아산지사2-2511", "N")

# 파단 전주 목록 (breakheight, breakdegree 포함)
break_poles = PDB.group_anal_type_pole_2("아산지사2-2511", "B")
# 반환: [{"poleid": "...", "breakheight": 1.5, "breakdegree": 90.0}, ...]
```

### 측정 데이터 조회
```python
# 측정 결과 정보
meas_result = PDB.get_meas_result("0432R025", "OUT")
# 반환: DataFrame (measno, stdegree, eddegree, stheight, edheight, ...)

# 측정 원시 데이터
meas_data_x = PDB.get_meas_data("0432R025", 1, "OUT", "x")
# 반환: DataFrame (idx, ch1, ch2, ..., ch8)
```

## 📌 참고사항

1. **프로젝트 이름 형식**: 대부분 `[지역명]-[날짜]` 형식이지만 일부는 다를 수 있음
2. **서버별 데이터**: 각 서버는 독립적인 데이터베이스를 가지고 있음
3. **데이터 동기화**: 서버 간 데이터 동기화는 수동으로 관리됨
4. **파단 정보**: 2차 분석(analstep=2) 결과만 파단 정보가 저장됨
5. **측정 데이터**: IN(내부)과 OUT(외부) 두 가지 타입의 측정 데이터가 있음

