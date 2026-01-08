# 파단 위치 정보 추출 가능 여부 분석

## 📊 현재 상황

### 1. 파단 위치 정보 저장 위치

#### A. `break_info.json` 파일
```json
{
  "poleid": "0621R481",
  "project_name": "강원동해-202209",
  "breakstate": "B",
  "breakheight": 0.9,      // 파단 높이 (미터 단위)
  "breakdegree": 130        // 파단 각도 (도 단위)
}
```

**특징**:
- ✅ 모든 파단 전주 폴더에 저장됨
- ⚠️ 일부 전주는 `null` 값일 수 있음 (예: 8732F191)

#### B. CSV 파일명
```
{전주ID}_{측정번호}_{날짜}_OUT_x_breakheight_{높이}_breakdegree_{각도}.csv
```

**특징**:
- ✅ 파일명에 파단 정보가 포함된 경우 추출 가능
- ⚠️ 일부 파일은 파단 정보가 파일명에 없을 수 있음

#### C. CSV 파일 내부 데이터
```csv
idx,groupname,poleid,devicetype,measno,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,axis
235034267,강릉지사-2506,8732F191,OUT,4,-0.0044,-0.0172,-0.0128,...
```

**특징**:
- ❌ CSV 파일 내부에는 **파단 위치 정보가 직접 포함되지 않음**
- ✅ 측정 데이터만 포함 (ch1~ch8 센서 값)

---

## 🔍 파단 위치 추출 방법

### 방법 1: `break_info.json` 파일에서 추출 (가장 확실)

```python
import json
import os

def extract_break_info_from_json(pole_dir):
    """전주 폴더에서 break_info.json 파일 읽기"""
    break_info_file = os.path.join(pole_dir, "*_break_info.json")
    json_files = glob.glob(break_info_file)
    
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            info = json.load(f)
            return info.get('breakheight'), info.get('breakdegree')
    return None, None

# 사용 예시
pole_dir = "raw_pole_data/break/강원동해-202209/0621R481"
breakheight, breakdegree = extract_break_info_from_json(pole_dir)
print(f"파단 높이: {breakheight}, 파단 각도: {breakdegree}")
```

**장점**:
- 가장 확실한 방법
- 모든 파단 전주에 존재
- 구조화된 데이터

**단점**:
- 일부 전주는 `null` 값일 수 있음

---

### 방법 2: CSV 파일명에서 추출

```python
import os
import re

def extract_break_info_from_filename(csv_file):
    """CSV 파일명에서 파단 정보 추출"""
    filename = os.path.basename(csv_file)
    
    # 패턴: _breakheight_{값}_breakdegree_{값}
    pattern = r'_breakheight_([\d.]+)_breakdegree_([\d.]+)'
    match = re.search(pattern, filename)
    
    if match:
        breakheight = float(match.group(1))
        breakdegree = float(match.group(2))
        return breakheight, breakdegree
    return None, None

# 사용 예시
csv_file = "8732F191_1_2025-07-16_OUT_x_breakheight_0.9_breakdegree_130.csv"
breakheight, breakdegree = extract_break_info_from_filename(csv_file)
```

**장점**:
- 파일명만으로 추출 가능
- 빠른 처리

**단점**:
- 파일명에 파단 정보가 없는 경우 추출 불가
- 일관성 부족 (일부 파일만 포함)

---

### 방법 3: 데이터베이스에서 측정 범위 정보 활용

```python
import poledb as PDB

def get_measurement_range(poleid, measno):
    """데이터베이스에서 측정 범위 정보 조회"""
    re_out = PDB.get_meas_result(poleid, 'OUT')
    
    # 해당 측정번호의 정보 찾기
    meas_info = re_out[re_out['measno'] == measno].iloc[0]
    
    return {
        'stdegree': meas_info['stdegree'],    # 시작 각도
        'eddegree': meas_info['eddegree'],    # 종료 각도
        'stheight': meas_info['stheight'],     # 시작 높이
        'edheight': meas_info['edheight']      # 종료 높이
    }

# 파단 위치가 측정 범위 내에 있는지 확인
def is_break_in_range(breakheight, breakdegree, meas_range):
    """파단 위치가 측정 범위 내에 있는지 확인"""
    return (meas_range['stdegree'] <= breakdegree <= meas_range['eddegree'] and
            meas_range['stheight'] <= breakheight <= meas_range['edheight'])
```

**장점**:
- 측정 범위 정보 활용 가능
- 파단 위치가 해당 측정 범위 내에 있는지 확인 가능

**단점**:
- 데이터베이스 연결 필요
- 직접적인 파단 위치는 제공하지 않음

---

### 방법 4: CSV 데이터에서 파단 위치 인덱스 계산

```python
import pandas as pd

def calculate_break_index_in_data(csv_data, breakheight, meas_range):
    """측정 데이터에서 파단 위치의 인덱스 계산"""
    # 측정 범위 정보 필요 (stheight, edheight)
    stheight = meas_range['stheight']
    edheight = meas_range['edheight']
    total_rows = len(csv_data)
    
    # 파단 높이가 측정 범위 내의 어느 위치인지 계산
    if edheight > stheight:
        relative_position = (breakheight - stheight) / (edheight - stheight)
        break_index = int(total_rows * relative_position)
        return break_index
    return None

# 사용 예시
csv_data = pd.read_csv("OUT_x.csv")
meas_range = get_measurement_range(poleid, measno)
break_index = calculate_break_index_in_data(csv_data, breakheight, meas_range)

# 파단 위치 주변 데이터 추출 (±10% 범위)
window_size = int(len(csv_data) * 0.1)
start_idx = max(0, break_index - window_size)
end_idx = min(len(csv_data), break_index + window_size)
break_region_data = csv_data[start_idx:end_idx]
```

**장점**:
- CSV 데이터에서 직접 파단 구간 추출 가능
- AI 학습 시 파단 구간에 집중 가능

**단점**:
- 측정 범위 정보 필요
- 파단 위치 정보가 먼저 필요

---

## 🎯 추천 방법

### 우선순위 1: `break_info.json` 파일 활용
```python
def get_break_position(pole_dir):
    """파단 위치 정보 추출 (통합 함수)"""
    # 1순위: JSON 파일에서 추출
    break_info_file = os.path.join(pole_dir, "*_break_info.json")
    json_files = glob.glob(break_info_file)
    
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            info = json.load(f)
            breakheight = info.get('breakheight')
            breakdegree = info.get('breakdegree')
            
            if breakheight is not None and breakdegree is not None:
                return breakheight, breakdegree
    
    # 2순위: 파일명에서 추출
    csv_files = glob.glob(os.path.join(pole_dir, "*_OUT_x*.csv"))
    for csv_file in csv_files:
        breakheight, breakdegree = extract_break_info_from_filename(csv_file)
        if breakheight is not None and breakdegree is not None:
            return breakheight, breakdegree
    
    return None, None
```

### 우선순위 2: 데이터베이스에서 조회
```python
# break_info.json이 null인 경우
# 데이터베이스에서 직접 조회
break_data = PDB.group_anal_type_pole_2(project_name, 'B')
for item in break_data:
    if item.get('poleid') == poleid:
        return item.get('breakheight'), item.get('breakdegree')
```

---

## 📝 결론

### ✅ 추출 가능 여부

| 방법 | 가능 여부 | 신뢰도 | 비고 |
|------|----------|--------|------|
| `break_info.json` | ✅ 가능 | ⭐⭐⭐⭐⭐ | 가장 확실 |
| CSV 파일명 | ⚠️ 부분 가능 | ⭐⭐⭐ | 파일명에 있는 경우만 |
| CSV 파일 내부 | ❌ 불가능 | - | 직접 정보 없음 |
| 데이터베이스 | ✅ 가능 | ⭐⭐⭐⭐ | DB 연결 필요 |

### 💡 권장 사항

1. **기본 방법**: `break_info.json` 파일에서 추출
2. **보조 방법**: 파일명에서 추출 (JSON이 null인 경우)
3. **최종 방법**: 데이터베이스에서 직접 조회

### 🔧 구현 예시

```python
import json
import os
import glob
import re

def extract_break_position(pole_dir):
    """
    전주 폴더에서 파단 위치 정보 추출
    
    Args:
        pole_dir: 전주 데이터 폴더 경로
    
    Returns:
        tuple: (breakheight, breakdegree) 또는 (None, None)
    """
    # 방법 1: JSON 파일에서 추출
    break_info_file = os.path.join(pole_dir, "*_break_info.json")
    json_files = glob.glob(break_info_file)
    
    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                info = json.load(f)
                breakheight = info.get('breakheight')
                breakdegree = info.get('breakdegree')
                
                if breakheight is not None and breakdegree is not None:
                    return float(breakheight), float(breakdegree)
        except Exception as e:
            print(f"JSON 파일 읽기 오류: {e}")
    
    # 방법 2: CSV 파일명에서 추출
    csv_files = glob.glob(os.path.join(pole_dir, "*_OUT_x*.csv"))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        pattern = r'_breakheight_([\d.]+)_breakdegree_([\d.]+)'
        match = re.search(pattern, filename)
        
        if match:
            try:
                breakheight = float(match.group(1))
                breakdegree = float(match.group(2))
                return breakheight, breakdegree
            except ValueError:
                continue
    
    return None, None

# 사용 예시
pole_dir = "raw_pole_data/break/강원동해-202209/0621R481"
breakheight, breakdegree = extract_break_position(pole_dir)

if breakheight is not None and breakdegree is not None:
    print(f"파단 위치 - 높이: {breakheight}m, 각도: {breakdegree}°")
else:
    print("파단 위치 정보를 찾을 수 없습니다.")
```

---

## 🚀 AI 학습 활용 방안

### 1. 파단 위치를 피처로 추가
```python
# 시퀀스 데이터에 파단 위치 정보 추가
breakheight_norm = (breakheight - min_height) / (max_height - min_height)
breakdegree_norm = breakdegree / 360.0

# 시퀀스 데이터와 결합
enhanced_sequence = np.concatenate([
    sequence_data,
    np.tile([breakheight_norm, breakdegree_norm], (sequence_length, 1))
], axis=-1)
```

### 2. 파단 위치 중심 윈도우 추출
```python
# 파단 위치를 중심으로 시퀀스 추출
break_index = calculate_break_index(breakheight, meas_range)
window_start = max(0, break_index - sequence_length // 2)
window_end = min(len(data), break_index + sequence_length // 2)
break_centered_sequence = data[window_start:window_end]
```

### 3. Attention 가중치 적용
```python
# 파단 위치 근처에 더 높은 가중치 부여
attention_weights = create_break_aware_attention(break_index, sequence_length)
```

