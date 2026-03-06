#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3. raw_pole_data의 OUT 파일들을 x, y, z로 병합·보간하여 통일된 데이터로 생성"""

import os
import argparse
import json
import glob
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))

# plot_processed_csv_2d 모듈 임포트 (같은 디렉토리)
sys.path.insert(0, current_dir)
try:
    from plot_processed_csv_2d import plot_csv_2d
except ImportError:
    plot_csv_2d = None

# 보간 간격 설정
HEIGHT_STEP = 0.1  # 높이 보간 간격: 10cm (0.1m)
DEGREE_STEP = 5.0  # 각도 보간 간격: 5도

def interpolate_data_to_grid(df, meas_info, target_heights, target_degrees):
    """
    데이터를 높이와 각도 그리드로 보간
    merge_pole_data_files_all.py의 로직을 참고하여 구현
    
    Args:
        df: 원본 데이터 (각 행이 하나의 높이에서 측정된 8개 센서 데이터)
        meas_info: 측정 정보 딕셔너리
        target_heights: 목표 높이 배열 (10cm 간격)
        target_degrees: 목표 각도 배열 (5도 간격, 0~360도)
    
    Returns:
        DataFrame: 보간된 데이터 (height, degree, value)
    """
    if df.empty:
        return pd.DataFrame()
    
    stheight = meas_info.get('stheight')
    edheight = meas_info.get('edheight')
    stdegree = meas_info.get('stdegree')
    eddegree = meas_info.get('eddegree')
    
    if eddegree is None:
        eddegree = stdegree + 360.0
    
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
    if eddegree < stdegree:
        angle_range = (360 - stdegree) + eddegree
    else:
        angle_range = eddegree - stdegree
    
    # 측정 범위가 0이거나 매우 작은 경우 처리
    if angle_range < 1.0:
        angle_range = 360.0
        stdegree = 0.0
        eddegree = 360.0
    
    # 채널별 각도 계산 (측정 범위 내에서 끝점 제외 균등 분배)
    # 예: 0~90도 측정 범위 → ch1=10도, ch2=20도, ..., ch8=80도
    num_channels = len(channel_cols)
    sensor_angles_by_channel = {}
    for idx, ch_col in enumerate(channel_cols):
        if angle_range > 0 and num_channels > 0:
            k = idx + 1  # 1..N
            angle = stdegree + (angle_range * k / (num_channels + 1))
        else:
            angle = stdegree
        angle = angle % 360.0
        sensor_angles_by_channel[ch_col] = float(angle)
    
    # 각 측정의 기준값 계산 (아웃라이어 제거 후 평균)
    all_channel_values = []
    for ch_col in channel_cols:
        ch_values = df[ch_col].dropna().values
        all_channel_values.extend(ch_values.tolist())
    
    baseline_value = None
    if len(all_channel_values) > 10:
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
    
    # target_heights는 이미 측정 범위(stheight ~ edheight) 내에서만 생성되었으므로
    # 추가 범위 체크는 불필요하지만, 명확성을 위해 주석으로 표시
    min_h = min(stheight, edheight)
    max_h = max(stheight, edheight)
    
    for target_height in target_heights:
        # 측정 범위 확인 (이미 필터링되었지만 안전장치)
        if target_height < min_h or target_height > max_h:
            continue
        
        # 각 채널에 대해 높이 보간을 먼저 수행
        channel_value_at_height = {}
        for ch_col in channel_cols:
            ch_values = df[ch_col].values
            
            # 선형 보간 (높이 기준)
            if len(heights) > 1 and len(np.unique(heights)) > 1:
                try:
                    interp_func = interp1d(
                        heights, ch_values, kind='linear',
                        fill_value=np.nan, bounds_error=False
                    )
                    channel_value_at_height[ch_col] = float(interp_func(target_height))
                except:
                    channel_value_at_height[ch_col] = float(ch_values[0]) if len(ch_values) > 0 else np.nan
            else:
                channel_value_at_height[ch_col] = float(ch_values[0]) if len(ch_values) > 0 else np.nan
        
        # 각 채널의 각도와 값을 매핑
        channel_angle_value_map = {}
        for ch_col in channel_cols:
            channel_angle = sensor_angles_by_channel.get(ch_col)
            channel_value = channel_value_at_height.get(ch_col)
            if channel_angle is not None and channel_value is not None and not np.isnan(channel_value):
                channel_angle_value_map[ch_col] = {
                    'angle': float(channel_angle),
                    'value': float(channel_value)
                }
        
        # degree 그리드에 채널값을 배치 (target_degrees는 이미 측정 범위 내이므로 범위 체크 불필요)
        for target_degree in target_degrees:
            row_data = {'height': round(target_height, 1), 'degree': target_degree}
            target_deg_float = float(target_degree)
            
            # 측정 범위 내: 목표 각도에 가장 가까운 채널 찾기
            closest_channels = []
            
            for ch_col, info in channel_angle_value_map.items():
                channel_angle = info['angle']
                channel_value = info['value']
                
                # 각도 차이 계산 (360도 경계 고려)
                angle_diff = abs(target_deg_float - channel_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                closest_channels.append((angle_diff, ch_col, channel_value, channel_angle))
            
            if not closest_channels:
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
                        ch1_diff, ch1_col, ch1_value, ch1_angle = closest_channels[0]
                        ch2_diff, ch2_col, ch2_value, ch2_angle = closest_channels[1]
                        
                        # 두 채널의 각도 차이 계산 (360도 경계 고려)
                        angle_between = abs(ch1_angle - ch2_angle)
                        if angle_between > 180:
                            angle_between = 360 - angle_between
                        
                        # 목표 각도와 각 채널의 각도 차이
                        diff_to_ch1 = abs(target_deg_float - ch1_angle)
                        if diff_to_ch1 > 180:
                            diff_to_ch1 = 360 - diff_to_ch1
                        diff_to_ch2 = abs(target_deg_float - ch2_angle)
                        if diff_to_ch2 > 180:
                            diff_to_ch2 = 360 - diff_to_ch2
                        
                        # 두 채널 사이에 있고, 각도 차이가 합리적이면 보간
                        if diff_to_ch1 + diff_to_ch2 <= angle_between * 1.5 and angle_between > 0:
                            total_diff = diff_to_ch1 + diff_to_ch2
                            if total_diff > 0:
                                weight_ch2 = diff_to_ch1 / total_diff
                                weight_ch1 = diff_to_ch2 / total_diff
                                interpolated_value = ch1_value * weight_ch1 + ch2_value * weight_ch2
                            else:
                                interpolated_value = ch1_value
                            row_data['value'] = interpolated_value
                        else:
                            row_data['value'] = closest_value
                    else:
                        row_data['value'] = closest_value
            
            final_data.append(row_data)
    
    if not final_data:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(final_data)
    return result_df


def process_out_files_for_measno(pole_dir, measno, meas_info, output_dir):
    """
    특정 측정 번호의 x, y, z OUT 파일을 읽어서 병합 및 보간 처리
    
    Args:
        pole_dir: 전주 디렉토리 경로
        measno: 측정 번호
        meas_info: 측정 정보 딕셔너리
        output_dir: 출력 디렉토리
    
    Returns:
        str or None: 저장된 파일 경로 또는 None
    """
    # 출력 파일 경로 결정
    poleid = os.path.basename(pole_dir)
    project_name = Path(pole_dir).parent.name
    
    output_subdir = os.path.join(output_dir, project_name, poleid)
    output_file = os.path.join(output_subdir, f"{poleid}_{measno}_OUT_processed.csv")
    
    # 이미 처리된 파일이 있으면 건너뛰기
    if os.path.exists(output_file):
        return output_file
    
    # 모든 OUT 파일 찾기 (파일명 패턴 대신 파일 내용의 measno로 찾기)
    all_out_files_x = glob.glob(os.path.join(pole_dir, "*OUT*x*.csv"))
    all_out_files_y = glob.glob(os.path.join(pole_dir, "*OUT*y*.csv"))
    all_out_files_z = glob.glob(os.path.join(pole_dir, "*OUT*z*.csv"))
    
    # 파일 내용의 measno로 올바른 파일 찾기
    file_x = None
    file_y = None
    file_z = None
    
    # X 파일 찾기
    for fx in all_out_files_x:
        try:
            df_check = pd.read_csv(fx)
            if not df_check.empty and 'measno' in df_check.columns:
                unique_measno = df_check['measno'].unique()
                if len(unique_measno) == 1 and unique_measno[0] == measno:
                    file_x = fx
                    break
        except:
            continue
    
    # Y 파일 찾기
    for fy in all_out_files_y:
        try:
            df_check = pd.read_csv(fy)
            if not df_check.empty and 'measno' in df_check.columns:
                unique_measno = df_check['measno'].unique()
                if len(unique_measno) == 1 and unique_measno[0] == measno:
                    file_y = fy
                    break
        except:
            continue
    
    # Z 파일 찾기
    for fz in all_out_files_z:
        try:
            df_check = pd.read_csv(fz)
            if not df_check.empty and 'measno' in df_check.columns:
                unique_measno = df_check['measno'].unique()
                if len(unique_measno) == 1 and unique_measno[0] == measno:
                    file_z = fz
                    break
        except:
            continue
    
    # 파일을 찾지 못한 경우 경고 메시지 출력
    if not file_x or not file_y or not file_z:
        missing = []
        if not file_x:
            missing.append("X")
        if not file_y:
            missing.append("Y")
        if not file_z:
            missing.append("Z")
        print(f"  경고 [{poleid}] measno {measno}: measno={measno}인 {', '.join(missing)} 파일을 찾을 수 없습니다. 스킵합니다.")
        return None
    
    # CSV 파일 읽기
    try:
        df_x = pd.read_csv(file_x)
        df_y = pd.read_csv(file_y)
        df_z = pd.read_csv(file_z)
    except Exception as e:
        print(f"  오류: CSV 파일 읽기 실패 - {e}")
        return None
    
    # 데이터가 비어있으면 스킵
    if df_x.empty or df_y.empty or df_z.empty:
        return None
    
    # 파일 내용의 measno 확인 (break_info.json의 measno와 일치하는지 검증)
    if 'measno' in df_x.columns:
        unique_measno_x = df_x['measno'].unique()
        if len(unique_measno_x) == 1 and unique_measno_x[0] != measno:
            # 파일 내용의 measno가 break_info.json의 measno와 일치하지 않으면 스킵
            print(f"  경고 [{poleid}] measno {measno}: break_info.json의 measno({measno})와 파일 내용의 measno({unique_measno_x[0]})가 일치하지 않습니다. 스킵합니다.")
            return None
    
    # 행 수가 동일한지 확인
    if len(df_x) != len(df_y) or len(df_x) != len(df_z):
        return None
    
    # 높이 범위 계산
    stheight = meas_info.get('stheight')
    edheight = meas_info.get('edheight')
    
    if stheight is None or edheight is None:
        return None
    
    min_height = min(stheight, edheight)
    max_height = max(stheight, edheight)
    
    # 목표 높이 배열 생성 (10cm 간격)
    if min_height == max_height:
        target_heights = np.array([min_height])
    else:
        target_heights = np.arange(min_height, max_height + HEIGHT_STEP, HEIGHT_STEP)
        target_heights = target_heights[(target_heights >= min_height) & (target_heights <= max_height)]
    
    # 목표 각도 배열 생성 (5도 간격, 측정 범위 내에서만)
    stdegree = meas_info.get('stdegree')
    eddegree = meas_info.get('eddegree')
    
    if stdegree is None or eddegree is None:
        return None
    
    # 각도 범위 계산
    if eddegree < stdegree:
        # 360도를 넘어가는 경우 (예: 270~90도)
        # stdegree부터 360도까지, 0도부터 eddegree까지
        target_degrees_1 = np.arange(stdegree, 360.0 + DEGREE_STEP, DEGREE_STEP)
        target_degrees_1 = target_degrees_1[target_degrees_1 <= 360.0]
        target_degrees_2 = np.arange(0.0, eddegree + DEGREE_STEP, DEGREE_STEP)
        target_degrees_2 = target_degrees_2[target_degrees_2 <= eddegree]
        target_degrees = np.concatenate([target_degrees_1, target_degrees_2])
        target_degrees = np.unique(target_degrees)
    else:
        # 일반적인 경우 (예: 0~90도)
        target_degrees = np.arange(stdegree, eddegree + DEGREE_STEP, DEGREE_STEP)
        target_degrees = target_degrees[(target_degrees >= stdegree) & (target_degrees <= eddegree)]
    
    # x, y, z 각각 보간 수행
    interpolated_x = interpolate_data_to_grid(df_x, meas_info, target_heights, target_degrees)
    interpolated_y = interpolate_data_to_grid(df_y, meas_info, target_heights, target_degrees)
    interpolated_z = interpolate_data_to_grid(df_z, meas_info, target_heights, target_degrees)
    
    if interpolated_x.empty or interpolated_y.empty or interpolated_z.empty:
        return None
    
    # x, y, z 데이터 병합
    # 모든 height, degree 조합 수집
    all_combinations = set()
    
    for df in [interpolated_x, interpolated_y, interpolated_z]:
        for _, row in df.iterrows():
            all_combinations.add((row['height'], row['degree']))
    
    # 결과 DataFrame 생성
    result_data = []
    
    for height, degree in sorted(all_combinations):
        row_data = {
            'height': height,
            'degree': degree,
            'x_value': np.nan,
            'y_value': np.nan,
            'z_value': np.nan
        }
        
        # x 값 찾기
        x_rows = interpolated_x[(interpolated_x['height'] == height) & (interpolated_x['degree'] == degree)]
        if not x_rows.empty:
            row_data['x_value'] = x_rows.iloc[0]['value']
        
        # y 값 찾기
        y_rows = interpolated_y[(interpolated_y['height'] == height) & (interpolated_y['degree'] == degree)]
        if not y_rows.empty:
            row_data['y_value'] = y_rows.iloc[0]['value']
        
        # z 값 찾기
        z_rows = interpolated_z[(interpolated_z['height'] == height) & (interpolated_z['degree'] == degree)]
        if not z_rows.empty:
            row_data['z_value'] = z_rows.iloc[0]['value']
        
        result_data.append(row_data)
    
    if not result_data:
        return None
    
    # DataFrame 생성
    output_df = pd.DataFrame(result_data)
    
    # 결측치가 있는 행도 포함 (필요에 따라 제거 가능)
    # output_df = output_df.dropna(subset=['x_value', 'y_value', 'z_value'])
    
    if output_df.empty:
        return None
    
    # devicetype 추가
    output_df['devicetype'] = 'OUT'
    
    # 정렬 (height, degree 순)
    output_df = output_df.sort_values(['height', 'degree']).reset_index(drop=True)
    
    # 필요한 컬럼만 선택
    output_df = output_df[['height', 'degree', 'x_value', 'y_value', 'z_value', 'devicetype']].copy()
    
    # 원본 보간된 값을 그대로 사용 (차분 계산하지 않음)
    # 모든 값이 NaN인 행 제거
    mask = ~(
        output_df['x_value'].isna() & 
        output_df['y_value'].isna() & 
        output_df['z_value'].isna()
    )
    output_df = output_df[mask].copy()
    
    # 필터링 후 데이터가 비어있으면 None 반환
    if output_df.empty:
        return None
    
    # 출력 디렉토리 생성
    os.makedirs(output_subdir, exist_ok=True)
    
    # 저장
    output_df.to_csv(output_file, index=False)
    
    return output_file


def process_pole_directory(pole_dir, output_dir):
    """
    한 전주 디렉토리의 모든 OUT 파일 처리
    파단 데이터의 경우 breakheight와 breakdegree가 모두 있을 때만 처리
    
    Args:
        pole_dir: 전주 디렉토리 경로
        output_dir: 출력 디렉토리
    
    Returns:
        int: 처리된 파일 개수
    """
    # break_info.json 또는 normal_info.json 파일 읽기
    poleid = os.path.basename(pole_dir)
    project_name = Path(pole_dir).parent.name
    info_file = os.path.join(pole_dir, f"{poleid}_break_info.json")
    is_break_info = True
    
    if not os.path.exists(info_file):
        # normal_info.json 시도
        info_file = os.path.join(pole_dir, f"{poleid}_normal_info.json")
        is_break_info = False
        if not os.path.exists(info_file):
            return 0
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            break_info = json.load(f)
    except Exception as e:
        return 0
    
    # 파단 데이터(breakstate='B')인 경우 breakheight와 breakdegree가 모두 있어야 함
    breakstate = break_info.get('breakstate')
    if breakstate == 'B':
        breakheight = break_info.get('breakheight')
        breakdegree = break_info.get('breakdegree')
        
        # breakheight와 breakdegree가 모두 없으면 처리하지 않음
        if breakheight is None or breakdegree is None:
            return 0
        
        # 숫자 타입 확인
        try:
            breakheight = float(breakheight)
            breakdegree = float(breakdegree)
        except (TypeError, ValueError):
            return 0
    
    measurements = break_info.get('measurements', {})
    
    # 전주 단위 확인: 모든 OUT measno에 대한 처리된 파일이 있는지 확인
    output_subdir = os.path.join(output_dir, project_name, poleid)
    all_processed = True
    out_measnos = []
    
    for meas_key, meas_info in measurements.items():
        if meas_info.get('devicetype') != 'OUT':
            continue
        
        measno = meas_info.get('measno')
        if measno is None:
            continue
        
        out_measnos.append(measno)
        output_file = os.path.join(output_subdir, f"{poleid}_{measno}_OUT_processed.csv")
        
        if not os.path.exists(output_file):
            all_processed = False
            break
    
    # 모든 OUT measno에 대한 처리된 파일이 있으면 전주 전체 건너뛰기
    if all_processed and len(out_measnos) > 0:
        return len(out_measnos)
    
    processed_count = 0
    created_csv_files = []  # 생성된 CSV 파일 목록
    
    # 각 OUT 측정 데이터 처리
    for meas_key, meas_info in measurements.items():
        if meas_info.get('devicetype') != 'OUT':
            continue
        
        measno = meas_info.get('measno')
        if measno is None:
            continue
        
        try:
            result = process_out_files_for_measno(pole_dir, measno, meas_info, output_dir)
            if result:
                created_csv_files.append(result)
        except Exception:
            continue
    
    # 파단 위치 검증 없이 모든 생성된 CSV 파일 유지 (예측용)
    processed_count = len(created_csv_files)
    
    # 파단 데이터이고 파일이 생성된 경우, 각 CSV 파일마다 별도의 break_info.json 저장 및 이미지 생성
    # 정상 데이터는 정보 파일을 저장하지 않음
    if processed_count > 0 and is_break_info and breakstate == 'B':
        project_name = Path(pole_dir).parent.name
        output_subdir = os.path.join(output_dir, project_name, poleid)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 각 CSV 파일마다 별도의 파단 정보 파일 생성
        for csv_file in created_csv_files:
            # CSV 파일명에서 measno 추출
            csv_basename = os.path.basename(csv_file)
            # 파일명 패턴: {poleid}_{measno}_OUT_processed.csv
            if csv_basename.endswith('_OUT_processed.csv'):
                prefix = csv_basename[:-len('_OUT_processed.csv')]
                # prefix는 {poleid}_{measno} 형태
                
                # 각 CSV 파일에 대응하는 파단 정보 파일 생성
                output_info_file = os.path.join(output_subdir, f"{prefix}_OUT_processed_break_info.json")
                
                try:
                    # 각 CSV 파일마다 동일한 파단 위치 정보를 저장 (처음에는 같은 위치)
                    csv_break_info = {
                        'poleid': poleid,
                        'project_name': project_name,
                        'breakstate': 'B',
                        'breakheight': breakheight,
                        'breakdegree': breakdegree,
                        'confirmed': False
                    }
                    
                    with open(output_info_file, 'w', encoding='utf-8') as f:
                        json.dump(csv_break_info, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass  # 저장 실패해도 무시
        
        # 생성된 CSV 파일들에 대해 이미지 생성 (각 CSV 파일에 대응하는 파단 정보 파일 사용)
        if plot_csv_2d is not None:
            for csv_file in created_csv_files:
                csv_basename = os.path.basename(csv_file)
                if csv_basename.endswith('_OUT_processed.csv'):
                    prefix = csv_basename[:-len('_OUT_processed.csv')]
                    output_info_file = os.path.join(output_subdir, f"{prefix}_OUT_processed_break_info.json")
                    
                    try:
                        plot_csv_2d(csv_file, None, output_info_file)
                    except Exception:
                        pass  # 이미지 생성 실패해도 무시
    
    # 정상 데이터는 정보 파일을 저장하지 않고 CSV 파일만 저장
    return processed_count


def process_all_raw_pole_data(
    raw_data_base_dir: str = "3. raw_pole_data",
    output_base_dir: str = "4. merge_data",
    normal_ratio: int = 10,
):
    """
    raw_pole_data 디렉토리 아래의 파단(break) 데이터를 머지하여 저장하고,
    정상(normal) 데이터는 이미지·파단 정보 없이 CSV만 합성하여
    파단 데이터 CSV 개수의 normal_ratio배 수만 랜덤 샘플로 저장한다.
    
    Args:
        raw_data_base_dir: 원본 데이터 기본 디렉토리
        output_base_dir: 출력 기본 디렉토리
        normal_ratio: 정상 샘플 비율(파단 대비 배수)
    """
    raw_data_path = Path(current_dir) / raw_data_base_dir
    
    if not raw_data_path.exists():
        print(f"[오류] 원본 데이터 디렉터리를 찾을 수 없습니다: {raw_data_base_dir}")
        return
    
    total_break_processed = 0
    
    # 1) 파단(break) 머지
    data_type = "break"
    data_type_path = raw_data_path / data_type
    if data_type_path.exists():
        output_path = Path(current_dir) / output_base_dir / data_type
        print(f"\n[정보][파단] 처리 시작")

        projects = [d for d in data_type_path.iterdir() if d.is_dir()]
        total_poles = 0
        total_processed = 0

        project_pbar = tqdm(sorted(projects), desc="  [파단] 프로젝트 처리", unit="프로젝트", leave=False)
        for project_dir in project_pbar:
            project_name = project_dir.name
            project_pbar.set_postfix_str(f"{project_name}", refresh=False)

            pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
            total_poles_in_project = len(pole_dirs)

            project_output_dir = output_path / project_name
            saved_pole_count = 0
            if project_output_dir.exists() and project_output_dir.is_dir():
                for pole_dir in project_output_dir.iterdir():
                    if pole_dir.is_dir() and list(pole_dir.glob("*_OUT_processed.csv")):
                        saved_pole_count += 1

            if saved_pole_count > 0 and saved_pole_count == total_poles_in_project:
                total_poles += total_poles_in_project
                for pole_dir in sorted(pole_dirs):
                    pole_output_dir = project_output_dir / pole_dir.name
                    if pole_output_dir.exists():
                        total_processed += len(list(pole_output_dir.glob("*_OUT_processed.csv")))
                continue

            for pole_dir in sorted(pole_dirs):
                total_poles += 1
                try:
                    processed_count = process_pole_directory(str(pole_dir), str(output_path))
                    if processed_count > 0:
                        total_processed += processed_count
                except Exception:
                    continue

        project_pbar.close()
        total_break_processed = total_processed
        print(f"\n[정보][파단] 완료: 전주 {total_poles}개, 파일 {total_processed}개")
    
    # 2) 정상(normal) 합성: 이미지·파단 정보는 생성하지 않고, 파단 CSV 개수의 10배만 랜덤 샘플로 저장
    normal_path = raw_data_path / "normal"
    normal_output_path = Path(current_dir) / output_base_dir / "normal"
    target_normal_count = normal_ratio * total_break_processed
    
    if normal_path.exists() and target_normal_count > 0:
        print(f"\n[정보][정상] 데이터 합성 시작 (목표: {target_normal_count}개)")
        
        all_normal_pole_dirs = []
        for project_dir in normal_path.iterdir():
            if not project_dir.is_dir():
                continue
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                info_file = pole_dir / f"{pole_dir.name}_normal_info.json"
                if info_file.exists():
                    all_normal_pole_dirs.append(pole_dir)
        
        random.shuffle(all_normal_pole_dirs)
        normal_processed = 0
        normal_pbar = tqdm(all_normal_pole_dirs, desc="  [정상] 전주 처리", unit="전주", leave=False)
        
        for pole_dir in normal_pbar:
            if normal_processed >= target_normal_count:
                break
            # 이미 있으면 건너뛰기: process_pole_directory와 동일한 경로 규칙 사용 (os.path.join)
            project_name = pole_dir.parent.name
            poleid = pole_dir.name
            output_pole_dir = os.path.join(str(normal_output_path), project_name, poleid)
            if os.path.exists(output_pole_dir):
                existing_csvs = glob.glob(os.path.join(output_pole_dir, "*_OUT_processed.csv"))
                if existing_csvs:
                    normal_processed += len(existing_csvs)
                    if normal_processed >= target_normal_count:
                        break
                    continue
            try:
                cnt = process_pole_directory(str(pole_dir), str(normal_output_path))
                normal_processed += cnt
            except Exception:
                continue
        
        normal_pbar.close()
        print(f"\n[정보][정상] 완료: {normal_processed}개 저장")
    
    print(f"\n[정보] 전체 처리 완료: {output_base_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="raw_pole_data를 merge_data 형식으로 변환")
    parser.add_argument("--raw-data-dir", default="3. raw_pole_data", help="입력 raw_pole_data 디렉터리")
    parser.add_argument("--output-dir", default="4. merge_data", help="출력 merge_data 디렉터리")
    parser.add_argument("--normal-ratio", type=int, default=10, help="정상 샘플 비율(파단 대비 배수)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("원본 데이터 처리 시작")
    print("=" * 60)

    process_all_raw_pole_data(
        raw_data_base_dir=args.raw_data_dir,
        output_base_dir=args.output_dir,
        normal_ratio=args.normal_ratio,
    )

    print("\n" + "=" * 60)
    print("원본 데이터 처리 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
