#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정상 전주의 여러 CSV 파일(OUT, x, y, z)을 하나의 파일로 합치는 스크립트
OUT 데이터만 처리 (IN 제외)

사용법:
    python merge_pole_data_files_normal.py
"""

import sys
import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d, griddata

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# merge_pole_data_files_all.py의 함수들 import
import importlib.util
merge_all_spec = importlib.util.spec_from_file_location(
    "merge_pole_data_files_all",
    os.path.join(current_dir, "4. merge_pole_data_files_all.py")
)
merge_all_module = importlib.util.module_from_spec(merge_all_spec)
merge_all_spec.loader.exec_module(merge_all_module)

merge_axis_data = merge_all_module.merge_axis_data
get_measurement_range_from_db = merge_all_module.get_measurement_range_from_db
HEIGHT_STEP = merge_all_module.HEIGHT_STEP
DEGREE_STEP = merge_all_module.DEGREE_STEP
OUT_DEGREE_MIN = merge_all_module.OUT_DEGREE_MIN
OUT_DEGREE_MAX = merge_all_module.OUT_DEGREE_MAX

# plot_merged_pole_data_contour.py의 함수 import
plot_contour_spec = importlib.util.spec_from_file_location(
    "plot_merged_pole_data_contour",
    os.path.join(current_dir, "plot_merged_pole_data_contour.py")
)
plot_contour_module = importlib.util.module_from_spec(plot_contour_spec)
plot_contour_spec.loader.exec_module(plot_contour_module)
plot_contour_2d = plot_contour_module.plot_contour_2d

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

# ============================================================================
# 설정
# ============================================================================
# 원본 정상 전주 데이터가 저장된 위치:
#   make_ai/3. raw_pole_data/normal/프로젝트명/전주ID/*.csv
INPUT_BASE_DIR = "3. raw_pole_data/normal"  # 입력 폴더 (정상 전주 원본 데이터)

# 머지 결과를 저장할 위치:
#   make_ai/4. merge_pole_data/normal_merged/프로젝트명/전주ID/{poleid}_OUT_merged.csv
# → 이후 5. crop_normal_region_from_merged.py에서
#   기본값 merged_base_dir="4. merge_pole_data/normal_merged" 으로 사용
OUTPUT_BASE_DIR = "4. merge_pole_data/normal_merged"  # 출력 폴더 (머지 결과)

MERGE_METHOD = "separate"  # "separate": IN/OUT 분리, "combined": 하나로 합침

# ============================================================================

def load_normal_info_json(pole_dir):
    """
    normal_info.json 파일을 읽어서 측정 정보 반환
    
    Args:
        pole_dir: 전주 데이터 폴더 경로
    
    Returns:
        dict: normal_info.json 내용 또는 None
    """
    poleid = os.path.basename(pole_dir)
    normal_info_file = os.path.join(pole_dir, f"{poleid}_normal_info.json")
    
    if os.path.exists(normal_info_file):
        try:
            with open(normal_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    경고 [{poleid}]: normal_info.json 읽기 실패: {e}")
            return None
    return None

def get_measurement_range_from_normal_info(normal_info, measno, devicetype='OUT'):
    """
    normal_info.json에서 측정 범위 정보 조회
    
    Args:
        normal_info: normal_info.json 내용
        measno: 측정 번호
        devicetype: 측정 타입 ('IN' 또는 'OUT')
    
    Returns:
        dict: 측정 범위 정보 (stheight, edheight, stdegree, eddegree) 또는 None
    """
    if normal_info is None:
        return None
    
    measurements = normal_info.get('measurements', {})
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
    
    # normal_info.json 파일 읽기
    normal_info = load_normal_info_json(pole_dir)
    
    # break_info 형식으로 변환 (merge_axis_data가 break_info를 받으므로)
    # normal_info도 동일한 구조를 가지고 있으므로 그대로 사용 가능
    break_info = normal_info  # 함수 내부에서는 break_info로 통일하여 사용
    
    # 실제로 생성된 파일 수 추적
    files_created = 0
    
    # OUT 데이터만 합치기
    if MERGE_METHOD == "separate":
        # OUT 데이터 합치기
        out_merged = merge_axis_data(out_files, 'OUT', poleid, server, project_name, break_info)
        if out_merged is not None and not out_merged.empty:
            # 출력 폴더 생성 (파일을 저장하기 직전에)
            os.makedirs(output_dir, exist_ok=True)
            out_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged.csv")
            out_merged.to_csv(out_output_path, index=False)
            files_created += 1
            
            # 이미지 파일 생성 및 저장 (정상 전주는 break_info=None으로 전달)
            try:
                image_output_path = os.path.join(output_dir, f"{poleid}_OUT_merged_contour_2d.png")
                plot_contour_2d(out_output_path, image_output_path, None)  # 정상 전주는 break_info 없음
            except Exception:
                pass
    
    elif MERGE_METHOD == "combined":
        # combined 방식은 OUT만 처리
        merge_all_data = merge_all_module.merge_all_data
        out_merged = merge_all_data({}, out_files, poleid, server, project_name, break_info)
        if out_merged is not None and not out_merged.empty:
            os.makedirs(output_dir, exist_ok=True)
            combined_output_path = os.path.join(output_dir, f"{poleid}_merged.csv")
            out_merged.to_csv(combined_output_path, index=False)
            files_created += 1
            
            # 이미지 파일 생성 및 저장 (정상 전주는 break_info=None으로 전달)
            try:
                image_output_path = os.path.join(output_dir, f"{poleid}_merged_contour_2d.png")
                plot_contour_2d(combined_output_path, image_output_path, None)  # 정상 전주는 break_info 없음
            except Exception:
                pass
    
    # 파일이 생성되지 않았으면 False 반환
    return files_created > 0

def process_all_poles():
    """
    모든 정상 전주 데이터 파일 합치기
    """
    input_dir = os.path.join(current_dir, INPUT_BASE_DIR)
    output_dir = os.path.join(current_dir, OUTPUT_BASE_DIR)
    
    if not os.path.exists(input_dir):
        print(f"오류: 입력 폴더를 찾을 수 없습니다: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 프로젝트 목록 가져오기
    projects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"전체 프로젝트 수: {len(projects)}개")
    print(f"합치기 방법: {MERGE_METHOD}")
    print(f"보간 간격: 높이 {HEIGHT_STEP*100}cm, 각도 {DEGREE_STEP}도")
    print(f"각도 범위: {OUT_DEGREE_MIN}~{OUT_DEGREE_MAX}도")
    
    # poledb 초기화
    if HAS_POLEDB:
        if PDB.poledb_conn is None:
            try:
                PDB.poledb_init()
                print("데이터베이스 연결 성공")
            except Exception as e:
                print(f"데이터베이스 연결 실패: {e}")
                return
    
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
            
            if merge_pole_data_files(pole_dir, pole_output_dir, None, project_name):
                success_count += 1
                project_success_count += 1
            else:
                if os.path.exists(pole_output_dir) and not os.listdir(pole_output_dir):
                    os.rmdir(pole_output_dir)
        
        if project_success_count == 0 and os.path.exists(project_output_dir) and not os.listdir(project_output_dir):
            os.rmdir(project_output_dir)
    
    print("\n" + "=" * 60)
    print("전체 처리 완료 (정상 전주)")
    print(f"처리된 전주 수: {total_poles}개")
    print(f"성공: {success_count}개")
    print(f"출력 위치: {output_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("정상 전주 데이터 파일 합치기 시작")
    print("=" * 60)
    
    try:
        process_all_poles()
        
        print("\n" + "=" * 60)
        print("정상 전주 데이터 파일 합치기 완료")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

