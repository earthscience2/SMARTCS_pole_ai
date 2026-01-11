#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한 전주의 여러 CSV 파일(OUT, x, y, z)을 하나의 파일로 합치는 스크립트
OUT 데이터만 처리 (IN 제외)

사용법:
    python merge_pole_data_files.py
"""

import sys
import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
import importlib.util

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# poledb 모듈 import
HAS_POLEDB = False
PDB = None

try:
    from config import poledb as PDB
    HAS_POLEDB = True
    print(f"poledb 모듈 import 성공 (경로: {parent_dir})")
except ImportError as e:
    HAS_POLEDB = False
    print(f"경고: poledb 모듈을 찾을 수 없습니다: {e}")
    print(f"검색 경로: {sys.path[:3]}")
    print("데이터베이스에서 측정 정보를 조회할 수 없습니다.")

# ============================================================================
# 설정
# ============================================================================
# 파단 및 정상 전주용 머지 스크립트:
#   - 입력:  make_ai/3. raw_pole_data/break/프로젝트명/전주ID/*.csv (파단)
#           make_ai/3. raw_pole_data/normal/프로젝트명/전주ID/*.csv (정상)
#   - 출력:  make_ai/4. merge_pole_data/break_merged/프로젝트명/전주ID/{poleid}_OUT_merged.csv (파단)
#           make_ai/4. merge_pole_data/normal_merged/프로젝트명/전주ID/{poleid}_OUT_merged.csv (정상)
INPUT_BASE_DIR_BREAK = "3. raw_pole_data/break"  # 입력 폴더 (파단 전주 원본 데이터)
INPUT_BASE_DIR_NORMAL = "3. raw_pole_data/normal"  # 입력 폴더 (정상 전주 원본 데이터)
OUTPUT_BASE_DIR_BREAK = "4. merge_pole_data/break_merged"  # 출력 폴더 (파단 머지 결과)
OUTPUT_BASE_DIR_NORMAL = "4. merge_pole_data/normal_merged"  # 출력 폴더 (정상 머지 결과)

MERGE_METHOD = "separate"  # "separate": IN/OUT 분리, "combined": 하나로 합침

# 보간 간격 설정
HEIGHT_STEP = 0.1  # 높이 보간 간격: 10cm (0.1m)
DEGREE_STEP = 5.0  # 각도 보간 간격: 5도

# OUT 데이터 각도 범위 (0~360도 전체)
OUT_DEGREE_MIN = 0.0
OUT_DEGREE_MAX = 360.0

# ============================================================================

def merge_pole_data_files(pole_dir, output_dir, server, project_name=None):
    """
    한 전주의 모든 CSV 파일을 합치기 (보간 적용)
    
    Args:
        pole_dir: 전주 데이터 폴더 경로
        output_dir: 출력 폴더 경로
        server: 서버 이름 (사용하지 않음)
        project_name: 프로젝트 이름 (데이터베이스 연결용)
    
    Returns:
        bool: 파일 생성 여부
    """
    if not os.path.exists(pole_dir):
        return False
    
    csv_files = glob.glob(os.path.join(pole_dir, "*.csv"))
    
    if not csv_files:
        return False
    
    # OUT 파일만 분류 (IN 제외)
    out_files = {
        'x': [],
        'y': [],
        'z': []
    }
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # 파일명에서 타입 추출 (OUT만)
        # 파일명 형식: {poleid}_{index}_{date}_OUT_{axis}_breakheight_{height}_breakdegree_{degree}.csv
        # 또는: {poleid}_{index}_{date}_OUT_{axis}.csv
        if '_OUT_' in filename and filename.endswith('.csv'):
            if '_OUT_x' in filename or filename.endswith('_OUT_x.csv'):
                out_files['x'].append(csv_file)
            elif '_OUT_y' in filename or filename.endswith('_OUT_y.csv'):
                out_files['y'].append(csv_file)
            elif '_OUT_z' in filename or filename.endswith('_OUT_z.csv'):
                out_files['z'].append(csv_file)
    
    # 파일명으로 정렬 (측정 번호 순서)
    for axis in out_files:
        out_files[axis].sort()
    
    poleid = os.path.basename(pole_dir)
    
    # break_info.json 또는 normal_info.json 파일 읽기
    pole_info = load_pole_info_json(pole_dir)
    
    # info.json 파일이 없으면 머지 진행하지 않음
    if pole_info is None:
        print(f"    경고 [{poleid}]: break_info.json 또는 normal_info.json이 없어서 머지를 건너뜁니다.")
        return False
    
    # breakstate 확인
    breakstate = pole_info.get('breakstate')
    
    # 파단 전주인 경우 추가 검증
    if breakstate == 'B':
        # breakheight와 breakdegree가 모두 유효한 값인지 확인
        breakheight = pole_info.get('breakheight')
        breakdegree = pole_info.get('breakdegree')
        
        # breakheight 유효성 검사
        is_breakheight_valid = False
        breakheight_float = None
        if breakheight is not None:
            try:
                breakheight_float = float(breakheight)
                if not (isinstance(breakheight_float, float) and np.isnan(breakheight_float)):
                    # 파단 높이가 3미터 이상이면 스킵
                    if breakheight_float < 3.0:
                        is_breakheight_valid = True
                    else:
                        print(f"    경고 [{poleid}]: breakheight({breakheight_float:.2f}m)가 3미터 이상이어서 머지를 건너뜁니다.")
            except (ValueError, TypeError):
                pass
        
        # breakdegree 유효성 검사
        is_breakdegree_valid = False
        breakdegree_float = None
        if breakdegree is not None:
            try:
                breakdegree_float = float(breakdegree)
                if not (isinstance(breakdegree_float, float) and np.isnan(breakdegree_float)):
                    # 각도가 0~360 사이인지 확인
                    if 0.0 <= breakdegree_float <= 360.0:
                        is_breakdegree_valid = True
                    else:
                        print(f"    경고 [{poleid}]: breakdegree({breakdegree_float:.1f}°)가 0~360 범위를 벗어나서 머지를 건너뜁니다.")
            except (ValueError, TypeError):
                pass
        
        # 둘 중 하나라도 유효하지 않으면 스킵
        if not is_breakheight_valid or not is_breakdegree_valid:
            if is_breakheight_valid and not is_breakdegree_valid:
                # breakheight는 유효하지만 breakdegree가 유효하지 않은 경우
                pass  # 이미 위에서 경고 메시지 출력됨
            elif not is_breakheight_valid and is_breakdegree_valid:
                # breakdegree는 유효하지만 breakheight가 유효하지 않은 경우
                pass  # 이미 위에서 경고 메시지 출력됨
            else:
                # 둘 다 유효하지 않은 경우
                print(f"    경고 [{poleid}]: breakheight({breakheight}) 또는 breakdegree({breakdegree})가 유효하지 않아서 머지를 건너뜁니다.")
            return False
    elif breakstate != 'N':
        # breakstate가 'B'도 'N'도 아닌 경우
        print(f"    경고 [{poleid}]: breakstate({breakstate})가 유효하지 않아서 머지를 건너뜁니다.")
        return False
    
    # 정상 전주는 검증 없이 진행
    
    # 이미 머지된 파일이 있는지 확인
    os.makedirs(output_dir, exist_ok=True)
    if MERGE_METHOD == "separate":
        out_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged.csv")
        if os.path.exists(out_output_path):
            print(f"    건너뛰기 [{poleid}]: 이미 머지된 파일이 존재합니다: {out_output_path}")
            return True  # 이미 존재하므로 성공으로 간주
    elif MERGE_METHOD == "combined":
        combined_output_path = os.path.join(output_dir, f"{poleid}_merged.csv")
        if os.path.exists(combined_output_path):
            print(f"    건너뛰기 [{poleid}]: 이미 머지된 파일이 존재합니다: {combined_output_path}")
            return True  # 이미 존재하므로 성공으로 간주
    
    # 실제로 생성된 파일 수 추적
    files_created = 0
    
    # plot_contour_2d 함수 import (필요할 때만)
    plot_contour_2d = None
    try:
        plot_contour_spec = importlib.util.spec_from_file_location(
            "plot_merged_pole_data_contour",
            os.path.join(current_dir, "plot_merged_pole_data_contour.py")
        )
        plot_contour_module = importlib.util.module_from_spec(plot_contour_spec)
        plot_contour_spec.loader.exec_module(plot_contour_module)
        plot_contour_2d = plot_contour_module.plot_contour_2d
    except Exception:
        pass
    
    # OUT 데이터만 합치기
    if MERGE_METHOD == "separate":
        # OUT 데이터 합치기
        out_merged = merge_axis_data(out_files, 'OUT', poleid, server, project_name, pole_info)
        if out_merged is not None and not out_merged.empty:
            # 출력 폴더 생성 (파일을 저장하기 직전에)
            os.makedirs(output_dir, exist_ok=True)
            out_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged.csv")
            out_merged.to_csv(out_output_path, index=False)
            files_created += 1
            
            # 이미지 파일 생성 및 저장
            if plot_contour_2d is not None:
                try:
                    image_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged_contour_2d.png")
                    plot_contour_2d(out_output_path, image_output_path, pole_info)
                except Exception:
                    pass
            
            # 파단 위치 메타데이터 파일 저장 (pole_info가 있고 breakstate가 'B'인 경우만)
            if pole_info is not None and pole_info.get('breakstate') == 'B':
                try:
                    metadata_output_path = os.path.join(output_dir, f"{poleid}_break_metadata.json")
                    image_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged_contour_2d.png")
                    metadata = {
                        'poleid': poleid,
                        'project_name': pole_info.get('project_name'),
                        'breakstate': pole_info.get('breakstate'),
                        'breakheight': pole_info.get('breakheight'),
                        'breakdegree': pole_info.get('breakdegree'),
                        'csv_file': os.path.basename(out_output_path),
                        'image_file': os.path.basename(image_output_path) if os.path.exists(image_output_path) else None,
                        'measurements_count': len(pole_info.get('measurements', {}))
                    }
                    with open(metadata_output_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    
    elif MERGE_METHOD == "combined":
        # combined 방식은 OUT만 처리
        out_merged = merge_all_data({}, out_files, poleid, server, project_name, pole_info)
        if out_merged is not None and not out_merged.empty:
            os.makedirs(output_dir, exist_ok=True)
            combined_output_path = os.path.join(output_dir, f"{poleid}_merged.csv")
            out_merged.to_csv(combined_output_path, index=False)
            files_created += 1
            
            # 이미지 파일 생성 및 저장
            if plot_contour_2d is not None:
                try:
                    image_output_path = os.path.join(output_dir, f"{poleid}_merged_contour_2d.png")
                    plot_contour_2d(combined_output_path, image_output_path, pole_info)
                except Exception:
                    pass
            
            # 파단 위치 메타데이터 파일 저장 (pole_info가 있고 breakstate가 'B'인 경우만)
            if pole_info is not None and pole_info.get('breakstate') == 'B':
                try:
                    metadata_output_path = os.path.join(output_dir, f"{poleid}_break_metadata.json")
                    image_output_path = os.path.join(output_dir, f"{poleid}_merged_contour_2d.png")
                    metadata = {
                        'poleid': poleid,
                        'project_name': pole_info.get('project_name'),
                        'breakstate': pole_info.get('breakstate'),
                        'breakheight': pole_info.get('breakheight'),
                        'breakdegree': pole_info.get('breakdegree'),
                        'csv_file': os.path.basename(combined_output_path),
                        'image_file': os.path.basename(image_output_path) if os.path.exists(image_output_path) else None,
                        'measurements_count': len(pole_info.get('measurements', {}))
                    }
                    with open(metadata_output_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    
    # 파일이 생성되지 않았으면 False 반환
    return files_created > 0

def load_pole_info_json(pole_dir):
    """
    break_info.json 또는 normal_info.json 파일을 읽어서 측정 정보 반환
    
    Args:
        pole_dir: 전주 데이터 폴더 경로
    
    Returns:
        dict: break_info.json 또는 normal_info.json 내용 또는 None
    """
    poleid = os.path.basename(pole_dir)
    
    # break_info.json 우선 확인
    break_info_file = os.path.join(pole_dir, f"{poleid}_break_info.json")
    if os.path.exists(break_info_file):
        try:
            import json
            with open(break_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    경고 [{poleid}]: break_info.json 읽기 실패: {e}")
            return None
    
    # normal_info.json 확인
    normal_info_file = os.path.join(pole_dir, f"{poleid}_normal_info.json")
    if os.path.exists(normal_info_file):
        try:
            import json
            with open(normal_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    경고 [{poleid}]: normal_info.json 읽기 실패: {e}")
            return None
    
    return None

def get_measurement_range_from_break_info(break_info, measno, devicetype='OUT'):
    """
    break_info.json에서 측정 범위 정보 조회
    
    Args:
        break_info: break_info.json 내용
        measno: 측정 번호
        devicetype: 측정 타입 ('IN' 또는 'OUT')
    
    Returns:
        dict: 측정 범위 정보 (stheight, edheight, stdegree, eddegree) 또는 None
    """
    if break_info is None:
        return None
    
    measurements = break_info.get('measurements', {})
    key = f"{devicetype}_{measno}"
    
    if key not in measurements:
        return None
    
    meas_info = measurements[key]
    
    return {
        'stheight': meas_info.get('stheight'),
        'edheight': meas_info.get('edheight'),
        'stdegree': meas_info.get('stdegree'),
        'eddegree': meas_info.get('eddegree')
    }

def apply_smoothing_to_result(result_df):
    """
    결과 데이터프레임에 2D 가우시안 필터를 적용하여 데이터를 부드럽게 만듦
    
    Args:
        result_df: 결과 데이터프레임 (height, degree, x_value, y_value, z_value)
    
    Returns:
        DataFrame: 스무딩이 적용된 데이터프레임
    """
    if result_df.empty:
        return result_df
    
    # 높이와 각도의 고유값 추출
    heights = sorted(result_df['height'].unique())
    degrees = sorted(result_df['degree'].unique())
    
    # 그리드 생성
    degree_grid, height_grid = np.meshgrid(degrees, heights)
    
    # 각 축별로 스무딩 적용
    for axis in ['x', 'y', 'z']:
        value_col = f'{axis}_value'
        if value_col not in result_df.columns:
            continue
        
        # 그리드 형태로 변환
        value_grid = np.full((len(heights), len(degrees)), np.nan)
        
        for _, row in result_df.iterrows():
            h = row['height']
            d = row['degree']
            
            h_idx = heights.index(h) if h in heights else None
            d_idx = degrees.index(d) if d in degrees else None
            
            if h_idx is not None and d_idx is not None:
                value_grid[h_idx, d_idx] = row[value_col] if pd.notna(row[value_col]) else np.nan
        
        # NaN이 아닌 값이 있는 경우에만 스무딩 적용
        if not np.all(np.isnan(value_grid)):
            # NaN을 0으로 임시 대체 (스무딩 후 원래 NaN 위치 복원)
            nan_mask = np.isnan(value_grid)
            value_grid_filled = np.where(nan_mask, 0, value_grid)
            
            # 2D 가우시안 필터 적용 (sigma=1.0으로 부드럽게)
            smoothed_grid = gaussian_filter(value_grid_filled, sigma=1.0, mode='nearest')
            
            # 원래 NaN 위치는 다시 NaN으로 복원
            smoothed_grid[nan_mask] = np.nan
            
            # 스무딩된 값을 다시 데이터프레임에 반영
            for idx, row in result_df.iterrows():
                h = row['height']
                d = row['degree']
                
                h_idx = heights.index(h) if h in heights else None
                d_idx = degrees.index(d) if d in degrees else None
                
                if h_idx is not None and d_idx is not None and not nan_mask[h_idx, d_idx]:
                    result_df.at[idx, value_col] = smoothed_grid[h_idx, d_idx]
    
    return result_df

def interpolate_data_to_grid(df, meas_range, target_heights, target_degrees):
    """
    데이터를 높이와 각도 그리드로 보간 (0~360도 전체 범위)
    
    Args:
        df: 원본 데이터 (각 행이 하나의 높이에서 측정된 8개 센서 데이터)
        meas_range: 측정 범위 정보
        target_heights: 목표 높이 배열 (10cm 간격)
        target_degrees: 목표 각도 배열 (5도 간격, 0~360도)
    
    Returns:
        DataFrame: 보간된 데이터 (height, degree, ch1~ch8)
    """
    if df.empty:
        return pd.DataFrame()
    
    stheight = meas_range['stheight']
    edheight = meas_range['edheight']
    stdegree = meas_range['stdegree']
    eddegree = meas_range['eddegree']
    
    # 원본 데이터의 높이 계산
    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame()
    
    # 각 행의 높이 계산 (선형 가정)
    relative_positions = np.linspace(0, 1, total_rows)
    if edheight != stheight:
        heights = stheight + relative_positions * (edheight - stheight)
    else:
        heights = np.full(total_rows, stheight)
    
    # 채널 데이터 선택 (ch1~ch8)
    channel_cols = [col for col in df.columns if col.startswith('ch') and col[2:].isdigit()]
    channel_cols = sorted(channel_cols, key=lambda x: int(x[2:]))
    
    if not channel_cols or len(channel_cols) != 8:
        return pd.DataFrame()
    
    # 각 측정의 각도 범위에서 8개 센서(채널) 각도 계산
    # 요구사항: 낮은 채널이 낮은 각도, 높은 채널이 높은 각도
    # 예) 0~90도를 측정했다면 ch1=10도, ch2=20도, ..., ch8=80도
    # => 범위의 양 끝(stdegree, eddegree)은 제외하고 (N+1) 등분점 중 1..N을 사용
    if eddegree < stdegree:
        angle_range = (360 - stdegree) + eddegree
    else:
        angle_range = eddegree - stdegree
    
    # 측정 범위가 0이거나 매우 작은 경우 처리 (기본값으로 0~360도 사용)
    if angle_range < 1.0:  # 1도 미만이면 전체 범위로 간주
        angle_range = 360.0
        stdegree = 0.0
        eddegree = 360.0
    
    # 채널별 각도(측정 범위 내에서 끝점 제외 균등 분배)
    # 예: 0~90도 측정 범위 → ch1=10도, ch2=20도, ..., ch8=80도
    num_channels = len(channel_cols)
    sensor_angles_by_channel = {}
    for idx, ch_col in enumerate(channel_cols):
        if angle_range > 0 and num_channels > 0:
            # idx=0..N-1 => k=1..N
            k = idx + 1
            angle = stdegree + (angle_range * k / (num_channels + 1))
        else:
            angle = stdegree
        angle = angle % 360.0
        
        # 실제 계산된 각도를 그대로 사용 (target_degrees에 맞추지 않음)
        sensor_angles_by_channel[ch_col] = float(angle)
    
    # 각 측정의 기준값 계산 (아웃라이어 제거 후 평균)
    # 모든 채널의 모든 값을 수집하여 기준값 계산
    all_channel_values = []
    for ch_col in channel_cols:
        ch_values = df[ch_col].dropna().values
        all_channel_values.extend(ch_values.tolist())
    
    baseline_value = None
    if len(all_channel_values) > 10:
        # 아웃라이어 제거 후 평균
        mean_value = np.mean(all_channel_values)
        distances = np.abs(np.array(all_channel_values) - mean_value)
        remove_count = max(1, int(len(all_channel_values) * 0.1))
        sorted_indices = np.argsort(distances)[::-1]
        remove_indices = sorted_indices[:remove_count]
        filtered_values = np.delete(np.array(all_channel_values), remove_indices)
        baseline_value = float(np.mean(filtered_values))
    elif len(all_channel_values) > 0:
        baseline_value = float(np.mean(all_channel_values))
    
    # 각 목표 높이와 각도 조합에 대해 모든 센서 데이터 보간
    final_data = []
    for target_height in target_heights:
        min_h = min(stheight, edheight)
        max_h = max(stheight, edheight)
        if target_height < min_h or target_height > max_h:
            continue
        
        # 각 채널에 대해 높이 보간을 먼저 수행
        channel_value_at_height = {}
        for ch_col in channel_cols:
            ch_values = df[ch_col].values

            # 선형 보간 (높이 기준)
            if len(heights) > 1 and len(np.unique(heights)) > 1:
                interp_func = interp1d(
                    heights, ch_values, kind='linear',
                    fill_value=np.nan, bounds_error=False
                )
                channel_value_at_height[ch_col] = float(interp_func(target_height))
            else:
                channel_value_at_height[ch_col] = float(ch_values[0]) if len(ch_values) > 0 else np.nan

        # 각 채널의 각도와 값을 매핑 (채널별 각도 정보 유지, 합치지 않음)
        channel_angle_value_map = {}
        for ch_col in channel_cols:
            channel_angle = sensor_angles_by_channel.get(ch_col)
            channel_value = channel_value_at_height.get(ch_col)
            if channel_angle is not None and channel_value is not None and not np.isnan(channel_value):
                channel_angle_value_map[ch_col] = {
                    'angle': float(channel_angle),
                    'value': float(channel_value)
                }
        
        # degree 그리드에 채널값을 배치: 각 채널의 각도에 해당하는 값만 사용
        for target_degree in target_degrees:
            row_data = {'height': round(target_height, 1), 'degree': target_degree}
            target_deg_float = float(target_degree)
            
            # 측정 범위 밖의 각도인지 확인
            # 각도 범위 계산 (360도 경계 고려)
            if eddegree < stdegree:
                # 360도를 넘어가는 경우 (예: 270~90도)
                is_in_range = (target_deg_float >= stdegree) or (target_deg_float <= eddegree)
            else:
                # 일반적인 경우 (예: 0~90도)
                is_in_range = (target_deg_float >= stdegree) and (target_deg_float <= eddegree)
            
            # 측정 범위 밖이면 기준값 사용
            if not is_in_range:
                if baseline_value is not None:
                    row_data['value'] = baseline_value
                else:
                    row_data['value'] = np.nan
                final_data.append(row_data)
                continue
            
            # 측정 범위 내: 목표 각도에 가장 가까운 채널 찾기
            closest_channels = []  # (angle_diff, ch_col, value, angle) 튜플 리스트
            
            for ch_col, info in channel_angle_value_map.items():
                channel_angle = info['angle']
                channel_value = info['value']
                
                # 각도 차이 계산 (360도 경계 고려)
                angle_diff = abs(target_deg_float - channel_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                closest_channels.append((angle_diff, ch_col, channel_value, channel_angle))
            
            if not closest_channels:
                # 채널이 없으면 기준값 사용
                if baseline_value is not None:
                    row_data['value'] = baseline_value
                else:
                    row_data['value'] = np.nan
            else:
                # 각도 차이로 정렬
                closest_channels.sort(key=lambda x: x[0])
                
                # 가장 가까운 채널
                min_angle_diff, closest_ch, closest_value, closest_angle = closest_channels[0]
                
                # 각도 차이가 매우 작으면 (2.5도 이내) 해당 채널의 값 직접 사용
                if min_angle_diff <= 2.5:
                    row_data['value'] = closest_value
                else:
                    # 두 채널 사이에 있으면 가장 가까운 두 채널로 선형 보간
                    if len(closest_channels) >= 2:
                        # 가장 가까운 두 채널 선택
                        ch1_diff, ch1_col, ch1_value, ch1_angle = closest_channels[0]
                        ch2_diff, ch2_col, ch2_value, ch2_angle = closest_channels[1]
                        
                        # 두 채널의 각도 차이 계산 (360도 경계 고려)
                        angle_between = abs(ch1_angle - ch2_angle)
                        if angle_between > 180:
                            angle_between = 360 - angle_between
                        
                        # 목표 각도가 두 채널 사이에 있는지 확인
                        # 목표 각도와 각 채널의 각도 차이
                        diff_to_ch1 = abs(target_deg_float - ch1_angle)
                        if diff_to_ch1 > 180:
                            diff_to_ch1 = 360 - diff_to_ch1
                        diff_to_ch2 = abs(target_deg_float - ch2_angle)
                        if diff_to_ch2 > 180:
                            diff_to_ch2 = 360 - diff_to_ch2
                        
                        # 두 채널 사이에 있고, 각도 차이가 합리적이면 보간
                        if diff_to_ch1 + diff_to_ch2 <= angle_between * 1.5 and angle_between > 0:
                            # 선형 보간 (두 채널만 사용)
                            # 각도 차이를 기준으로 가중치 계산
                            total_diff = diff_to_ch1 + diff_to_ch2
                            if total_diff > 0:
                                weight_ch2 = diff_to_ch1 / total_diff
                                weight_ch1 = diff_to_ch2 / total_diff
                                interpolated_value = ch1_value * weight_ch1 + ch2_value * weight_ch2
                            else:
                                interpolated_value = ch1_value
                            row_data['value'] = interpolated_value
                        else:
                            # 두 채널 사이에 있지 않으면 가장 가까운 채널의 값 사용
                            row_data['value'] = closest_value
                    else:
                        # 채널이 1개만 있으면 그 값 사용
                        row_data['value'] = closest_value
            
            final_data.append(row_data)
    
    if not final_data:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(final_data)
    return result_df

def merge_axis_data(file_dict, devicetype, poleid, server, project_name=None, break_info=None):
    """
    OUT 데이터의 x, y, z 데이터를 합치기 (높이와 각도 2차원 보간 적용)
    
    Args:
        file_dict: {'x': [...], 'y': [...], 'z': [...]} 형태의 파일 딕셔너리
        devicetype: 'OUT'
        poleid: 전주 ID
        server: 서버 이름
        project_name: 프로젝트 이름 (데이터베이스 연결용)
        break_info: break_info.json 내용 (선택사항)
    
    Returns:
        DataFrame: 합쳐진 데이터
    """
    all_measurements = {}  # 측정번호별로 데이터 저장
    
    # 각 축(x, y, z)별로 처리
    for axis in ['x', 'y', 'z']:
        files = file_dict[axis]
        
        if not files:
            continue
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                
                df = pd.read_csv(file_path)
                
                if df.empty:
                    continue
                
                # 파일명에서 측정 정보 추출
                parts = filename.replace('.csv', '').split('_')
                
                file_index = None
                if len(parts) >= 2:
                    try:
                        file_index = int(parts[1])
                    except (ValueError, IndexError):
                        continue
                
                if file_index is None:
                    continue
                
                # OUT 데이터만 처리
                if '_OUT_' not in filename:
                    continue
                
                file_devicetype = 'OUT'
                
                # 파일명의 인덱스를 실제 measno로 변환
                # break_info.json 우선, 없으면 DB에서 조회
                measno = None
                
                # break_info.json에서 먼저 조회
                if break_info is not None:
                    measurements = break_info.get('measurements', {})
                    # 방법 1: file_index를 키로 직접 조회 (OUT_1, OUT_2, ...)
                    key = f"{file_devicetype}_{file_index}"
                    if key in measurements:
                        measno = measurements[key].get('measno')
                        if measno is not None:
                            measno = int(measno)
                    else:
                        # 방법 2: file_index를 순서 인덱스로 사용하여 measurements에서 찾기
                        out_keys = [k for k in measurements.keys() if k.startswith(f"{file_devicetype}_")]
                        out_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
                        if file_index <= len(out_keys):
                            key = out_keys[file_index - 1]
                            measno = measurements[key].get('measno')
                            if measno is not None:
                                measno = int(measno)
                
                # measno를 찾지 못했으면 스킵 (break_info.json에서만 조회)
                if measno is None:
                    continue
                
                # 측정 범위 정보 조회 (break_info.json에서만 조회)
                meas_range = None
                
                if break_info is not None:
                    meas_range = get_measurement_range_from_break_info(break_info, measno, file_devicetype)
                
                # break_info.json에 측정 범위 정보가 없으면 스킵
                if meas_range is None:
                    continue
                
                # 측정 각도 차이 계산 (90도 이상일 경우에만 머지)
                stdegree = meas_range['stdegree']
                eddegree = meas_range['eddegree']
                
                # 각도 범위 계산
                if eddegree < stdegree:
                    angle_range = (360 - stdegree) + eddegree
                else:
                    angle_range = eddegree - stdegree
                
                # 측정 각도 차이가 90도 미만이면 스킵
                if abs(angle_range) < 90.0:
                    continue
                
                # 채널 데이터만 선택
                channel_cols = [col for col in df.columns if col.startswith('ch') and col[2:].isdigit()]
                
                if not channel_cols:
                    continue
                
                # 축 정보를 컬럼명에 추가 (x_ch1, y_ch1, ...)
                df_channels = df[channel_cols].copy()
                df_channels = df_channels.rename(columns={col: f'{axis}_{col}' for col in channel_cols})
                
                # 측정번호별로 데이터 저장
                if measno not in all_measurements:
                    all_measurements[measno] = {
                        'x': None,
                        'y': None,
                        'z': None,
                        'meas_range': meas_range
                    }
                
                all_measurements[measno][axis] = df_channels
                
            except Exception:
                continue
    
    if not all_measurements:
        return None
    
    # 모든 측정의 공통 높이 범위 계산 (합집합)
    all_stheights = [m['meas_range']['stheight'] for m in all_measurements.values()]
    all_edheights = [m['meas_range']['edheight'] for m in all_measurements.values()]
    
    # 전체 높이 범위 (합집합) - 항상 낮은 값부터 높은 값 순서로 정규화
    min_height = min(min(all_stheights), min(all_edheights))
    max_height = max(max(all_stheights), max(all_edheights))
    
    print(f"    디버깅 [{poleid}]: 공통 높이 범위 {min_height:.2f}~{max_height:.2f}m")
    
    if min_height == max_height:
        # 높이가 하나뿐인 경우
        target_heights = np.array([min_height])
    else:
        target_heights = np.arange(min_height, max_height + HEIGHT_STEP, HEIGHT_STEP)
        target_heights = target_heights[(target_heights >= min_height) & (target_heights <= max_height)]
    
    # 목표 각도 배열 생성 (5도 간격, 0~360도)
    target_degrees = np.arange(OUT_DEGREE_MIN, OUT_DEGREE_MAX + DEGREE_STEP, DEGREE_STEP)
    target_degrees = target_degrees[(target_degrees >= OUT_DEGREE_MIN) & (target_degrees <= OUT_DEGREE_MAX)]
    
    # 각 측정 데이터를 그리드로 보간
    all_interpolated = []
    
    for measno, meas_data in all_measurements.items():
        meas_range = meas_data['meas_range']
        
        # 각 축 데이터를 개별적으로 보간
        for axis in ['x', 'y', 'z']:
            axis_data = meas_data[axis]
            
            if axis_data is None or axis_data.empty:
                continue
            
            # 원본 데이터프레임 준비 (채널 컬럼명을 원래대로)
            original_cols = {f'{axis}_ch{i}': f'ch{i}' for i in range(1, 9)}
            axis_data_original = axis_data.rename(columns=original_cols)
            
            # 보간 수행
            try:
                interpolated = interpolate_data_to_grid(
                    axis_data_original, meas_range, target_heights, target_degrees
                )
                
                if not interpolated.empty:
                    # value 컬럼을 axis_value로 변경
                    if 'value' in interpolated.columns:
                        interpolated = interpolated.rename(columns={'value': f'{axis}_value'})
                    interpolated['measno'] = measno
                    interpolated['axis'] = axis
                    all_interpolated.append(interpolated)
            except Exception:
                continue
    
    if not all_interpolated:
        return None
    
    # 각 측정의 기준값 계산 (아웃라이어 제거 후 평균)
    # 기준값: 해당 데이터의 평균에서 많이 떨어진 상위 10% 값들을 제거한 값들의 평균
    baseline_values = {}  # (measno, axis) -> baseline_value
    
    for interp_df in all_interpolated:
        measno = interp_df['measno'].iloc[0] if 'measno' in interp_df.columns else None
        axis = interp_df['axis'].iloc[0] if 'axis' in interp_df.columns else None
        value_col = f'{axis}_value' if axis else 'value'
        
        if measno is None or axis is None or value_col not in interp_df.columns:
            continue
        
        # 유효한 값만 추출
        valid_values = interp_df[value_col].dropna().values
        
        if len(valid_values) == 0:
            continue
        
        # 아웃라이어 제거 후 평균 계산
        if len(valid_values) > 10:  # 값이 충분히 많을 때만 아웃라이어 제거
            # 평균 계산
            mean_value = np.mean(valid_values)
            
            # 평균에서의 거리 계산
            distances = np.abs(valid_values - mean_value)
            
            # 상위 10% (가장 멀리 떨어진 값들) 제거
            remove_count = max(1, int(len(valid_values) * 0.1))  # 최소 1개는 제거
            sorted_indices = np.argsort(distances)[::-1]  # 거리가 큰 순서로 정렬
            remove_indices = sorted_indices[:remove_count]
            
            # 아웃라이어 제거
            filtered_values = np.delete(valid_values, remove_indices)
            
            # 필터링된 값들의 평균을 기준값으로 사용
            baseline_value = np.mean(filtered_values)
        else:
            # 값이 적으면 전체 평균 사용
            baseline_value = np.mean(valid_values)
        
        baseline_values[(measno, axis)] = float(baseline_value)
    
    # 모든 보간 데이터 합치기
    # 높이와 각도 조합으로 그리드 생성
    all_height_degree_combinations = []
    for interp_df in all_interpolated:
        for _, row in interp_df.iterrows():
            h_d = (row['height'], row['degree'])
            if h_d not in all_height_degree_combinations:
                all_height_degree_combinations.append(h_d)
    
    # 정렬
    all_height_degree_combinations.sort()
    
    # 결과 데이터프레임 생성 (height, degree, x_value, y_value, z_value)
    result = pd.DataFrame(all_height_degree_combinations, columns=['height', 'degree'])
    
    # 겹치는 부분을 처리하기 위해 각 (height, degree) 조합에 대한 모든 변화량 수집
    # 딕셔너리: (height, degree) -> {'x': [delta_values...], 'y': [delta_values...], 'z': [delta_values...]}
    delta_collections = {}
    baseline_collections = {}  # 각 축별 기준값 수집 (평균 계산용)
    
    for interp_df in all_interpolated:
        measno = interp_df['measno'].iloc[0] if 'measno' in interp_df.columns else None
        axis = interp_df['axis'].iloc[0] if 'axis' in interp_df.columns else None
        value_col = f'{axis}_value' if axis else 'value'
        
        if measno is None or axis is None or value_col not in interp_df.columns:
            continue
        
        # 이 측정의 기준값 가져오기
        baseline_value = baseline_values.get((measno, axis))
        if baseline_value is None:
            # 기준값이 없으면 절대값 그대로 사용
            baseline_value = 0.0
        
        # 기준값 수집 (평균 계산용)
        if axis not in baseline_collections:
            baseline_collections[axis] = []
        if baseline_value != 0.0:  # 기준값이 0이 아닌 경우만 수집
            baseline_collections[axis].append(baseline_value)
        
        # 각 행의 (height, degree)와 변화량 수집
        for _, row in interp_df.iterrows():
            h = row['height']
            d = row['degree']
            value = row[value_col]
            
            if pd.notna(value):
                # 변화량 계산 (절대값 - 기준값)
                delta_value = float(value) - baseline_value
                
                key = (h, d)
                if key not in delta_collections:
                    delta_collections[key] = {'x': [], 'y': [], 'z': []}
                
                if axis in ['x', 'y', 'z']:
                    delta_collections[key][axis].append(delta_value)
    
    # 각 축별 평균 기준값 계산
    avg_baselines = {}
    for axis in ['x', 'y', 'z']:
        if axis in baseline_collections and baseline_collections[axis]:
            avg_baselines[axis] = np.mean(baseline_collections[axis])
        else:
            avg_baselines[axis] = 0.0
    
    # 수집된 변화량들을 평균값으로 계산하고, 평균 기준값을 더해서 절대값으로 복원
    # 중복되는 영역의 경우 여러 측정의 변화량을 평균하여 사용
    result['x_value'] = np.nan
    result['y_value'] = np.nan
    result['z_value'] = np.nan
    
    overlap_count = 0  # 중복 영역 카운트
    
    for idx, result_row in result.iterrows():
        h = result_row['height']
        d = result_row['degree']
        key = (h, d)
        
        if key in delta_collections:
            # 각 축별로 변화량의 평균 계산 후 평균 기준값 더하기
            for axis in ['x', 'y', 'z']:
                delta_values = delta_collections[key][axis]
                if delta_values:
                    # 중복 영역: 여러 측정이 겹치는 경우 변화량의 평균 사용
                    if len(delta_values) > 1:
                        overlap_count += 1
                    
                    # 변화량의 평균 계산
                    avg_delta = np.mean(delta_values)
                    
                    # 평균 기준값을 더해서 절대값으로 복원
                    absolute_value = avg_delta + avg_baselines[axis]
                    result.at[idx, f'{axis}_value'] = absolute_value
    
    result['devicetype'] = devicetype
    
    # 높이 값을 소수점 한자리로 반올림
    if 'height' in result.columns:
        result['height'] = result['height'].round(1)
    
    # 데이터 스무딩 적용 (2D 가우시안 필터)
    result = apply_smoothing_to_result(result)
    
    return result

def merge_all_data(in_files, out_files, poleid, server, project_name=None, break_info=None):
    """
    OUT 데이터만 합치기 (IN 데이터 제외)
    """
    out_data = merge_axis_data(out_files, 'OUT', poleid, server, project_name, break_info)
    return out_data

def process_all_poles():
    """
    모든 전주 데이터 파일 합치기 (파단 및 정상 전주 모두 처리)
    """
    # 파단 전주와 정상 전주 모두 처리
    input_dirs = [
        (os.path.join(current_dir, INPUT_BASE_DIR_BREAK), os.path.join(current_dir, OUTPUT_BASE_DIR_BREAK), "파단"),
        (os.path.join(current_dir, INPUT_BASE_DIR_NORMAL), os.path.join(current_dir, OUTPUT_BASE_DIR_NORMAL), "정상")
    ]
    
    print(f"합치기 방법: {MERGE_METHOD}")
    print(f"보간 간격: 높이 {HEIGHT_STEP*100}cm, 각도 {DEGREE_STEP}도")
    print(f"각도 범위: {OUT_DEGREE_MIN}~{OUT_DEGREE_MAX}도")
    print("주의: break_info.json 또는 normal_info.json에서만 측정 정보를 가져옵니다 (DB 접속 없음)")
    
    total_poles_all = 0
    success_count_all = 0
    
    for input_dir, output_dir, category in input_dirs:
        if not os.path.exists(input_dir):
            print(f"\n경고: {category} 전주 입력 폴더를 찾을 수 없습니다: {input_dir}")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 프로젝트 목록 가져오기
        projects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
        print(f"\n{'='*60}")
        print(f"[{category} 전주] 전체 프로젝트 수: {len(projects)}개")
        print(f"{'='*60}")
        
        total_poles = 0
        success_count = 0
        
        for project_idx, project_name in enumerate(projects, 1):
            project_dir = os.path.join(input_dir, project_name)
            pole_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
            
            print(f"\n[{project_idx}/{len(projects)}] 프로젝트: {project_name}")
            print(f"  전주 수: {len(pole_dirs)}개")
            
            project_output_dir = os.path.join(output_dir, project_name)
            project_success_count = 0
            
            for poleid in tqdm(pole_dirs, desc=f"  {project_name} 처리 중"):
                pole_dir = os.path.join(project_dir, poleid)
                pole_output_dir = os.path.join(project_output_dir, poleid)
                
                total_poles += 1
                total_poles_all += 1
                
                if merge_pole_data_files(pole_dir, pole_output_dir, None, project_name):
                    success_count += 1
                    success_count_all += 1
                    project_success_count += 1
                else:
                    if os.path.exists(pole_output_dir) and not os.listdir(pole_output_dir):
                        os.rmdir(pole_output_dir)
            
            if project_success_count == 0 and os.path.exists(project_output_dir) and not os.listdir(project_output_dir):
                os.rmdir(project_output_dir)
        
        print(f"\n[{category} 전주] 처리 완료")
        print(f"  처리된 전주 수: {total_poles}개")
        print(f"  성공: {success_count}개")
        print(f"  출력 위치: {output_dir}")
    
    print("\n" + "=" * 60)
    print("전체 처리 완료")
    print(f"전체 처리된 전주 수: {total_poles_all}개")
    print(f"전체 성공: {success_count_all}개")
    print(f"  - 파단 전주 출력: {OUTPUT_BASE_DIR_BREAK}")
    print(f"  - 정상 전주 출력: {OUTPUT_BASE_DIR_NORMAL}")

if __name__ == "__main__":
    print("=" * 60)
    print("전주 데이터 파일 합치기 시작")
    print("=" * 60)
    
    try:
        process_all_poles()
        
        print("\n" + "=" * 60)
        print("전주 데이터 파일 합치기 완료")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
