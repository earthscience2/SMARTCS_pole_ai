#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
break_info.json에서 파단 위치(breakheight, breakdegree)가 포함된 CSV 파일만 선택하여
5. select_pole_data 디렉토리로 복사

각 전주 폴더에서:
- break_info.json의 breakheight와 breakdegree 값이 CSV 파일에 포함되어 있는 파일만 선택
- 선택된 CSV 파일과 break_info.json을 5. select_pole_data 디렉토리로 복사 (디렉토리 구조 유지)
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_break_info(break_info_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    break_info.json에서 breakheight와 breakdegree 읽기.
    
    Args:
        break_info_path: break_info.json 파일 경로
    
    Returns:
        tuple: (breakheight, breakdegree) 또는 (None, None)
    """
    if not os.path.exists(break_info_path):
        return None, None

    try:
        with open(break_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception as e:
        print(f"  경고: break_info.json 읽기 실패 - {e}")
        return None, None

    bh = info.get("breakheight")
    bd = info.get("breakdegree")

    if bh is None:
        return None, None

    try:
        break_height = float(bh)
        break_degree = float(bd) if bd is not None else None
        return break_height, break_degree
    except (ValueError, TypeError):
        return None, None


def csv_contains_break_location(
    csv_path: str,
    break_height: float,
    break_degree: Optional[float],
    height_tolerance: float = 0.01,
    degree_tolerance: float = 0.5
) -> bool:
    """
    CSV 파일에 파단 위치(breakheight, breakdegree)가 포함되어 있는지 확인.
    
    Args:
        csv_path: CSV 파일 경로
        break_height: 파단 높이
        break_degree: 파단 각도 (None이면 높이만 확인)
        height_tolerance: 높이 허용 오차 (기본값: 0.01m = 1cm)
        degree_tolerance: 각도 허용 오차 (기본값: 0.5도)
    
    Returns:
        bool: 파단 위치가 포함되어 있으면 True
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return False
        
        # 필요한 컬럼 확인
        if 'height' not in df.columns:
            return False
        
        # 높이 확인 (breakheight가 CSV의 height 범위에 있는지)
        height_min = df['height'].min()
        height_max = df['height'].max()
        if not (height_min <= break_height <= height_max):
            return False
        
        # breakdegree가 None이면 높이만 확인
        if break_degree is None:
            # 높이 범위에 정확히 포함되는지 확인
            height_mask = np.abs(df['height'] - break_height) <= height_tolerance
            return height_mask.any()
        
        # breakdegree도 확인
        if 'degree' not in df.columns:
            return False
        
        # breakheight와 breakdegree 모두 포함되는지 확인
        height_mask = np.abs(df['height'] - break_height) <= height_tolerance
        degree_mask = np.abs(df['degree'] - break_degree) <= degree_tolerance
        
        # 두 조건을 모두 만족하는 행이 있는지 확인
        combined_mask = height_mask & degree_mask
        return combined_mask.any()
        
    except Exception as e:
        print(f"    경고: CSV 파일 확인 중 오류 ({csv_path}): {e}")
        return False


def process_pole_directory(pole_dir: Path, output_base_dir: Path) -> Tuple[int, int]:
    """
    한 전주 디렉토리 처리: 파단 위치가 포함된 CSV 파일만 복사.
    
    Args:
        pole_dir: 전주 디렉토리 경로
        output_base_dir: 출력 기본 디렉토리 (5. select_pole_data)
    
    Returns:
        tuple: (복사된 파일 수, 스킵된 파일 수)
    """
    poleid = pole_dir.name
    project_dir = pole_dir.parent
    project_name = project_dir.name
    
    # break_info.json 파일 찾기
    break_info_json = pole_dir / f"{poleid}_break_info.json"
    
    if not break_info_json.exists():
        return 0, 0
    
    # break_info.json에서 breakheight와 breakdegree 읽기
    break_height, break_degree = load_break_info(str(break_info_json))
    
    if break_height is None:
        return 0, 0
    
    # 모든 *_processed.csv 파일 찾기
    csv_files = list(pole_dir.glob("*_processed.csv"))
    
    if not csv_files:
        return 0, 0
    
    copied_count = 0
    skipped_count = 0
    
    # 출력 디렉토리 생성 (프로젝트/전주ID 구조 유지)
    output_pole_dir = output_base_dir / project_name / poleid
    output_pole_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_file in csv_files:
        # CSV 파일에 파단 위치가 포함되어 있는지 확인
        if csv_contains_break_location(str(csv_file), break_height, break_degree):
            # CSV 파일 복사
            output_csv_file = output_pole_dir / csv_file.name
            shutil.copy2(csv_file, output_csv_file)
            copied_count += 1
        else:
            skipped_count += 1
    
    # 파단 위치가 포함된 CSV 파일이 하나라도 있으면 break_info.json도 복사
    if copied_count > 0:
        output_break_info = output_pole_dir / break_info_json.name
        shutil.copy2(break_info_json, output_break_info)
    
    return copied_count, skipped_count


def process_all_poles(
    input_dir: str = "4. edit_pole_data/break",
    output_dir: str = "5. select_pole_data"
):
    """
    모든 전주 디렉토리를 처리하여 파단 위치가 포함된 CSV 파일만 선택하여 복사.
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: {input_dir}에서 프로젝트 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"총 프로젝트 수: {len(projects)}개")
    print("=" * 80)
    
    total_copied = 0
    total_skipped = 0
    processed_poles = 0
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        project_copied = 0
        project_skipped = 0
        
        for pole_dir in pole_dirs:
            poleid = pole_dir.name
            
            copied, skipped = process_pole_directory(pole_dir, output_path)
            
            if copied > 0:
                project_copied += copied
                project_skipped += skipped
                processed_poles += 1
                print(f"  [{poleid}] 복사: {copied}개, 스킵: {skipped}개")
        
        if project_copied > 0:
            print(f"  → 프로젝트 합계: 복사 {project_copied}개, 스킵 {project_skipped}개")
        
        total_copied += project_copied
        total_skipped += project_skipped
    
    print("\n" + "=" * 80)
    print(f"완료!")
    print(f"  처리된 전주 수: {processed_poles}개")
    print(f"  총 복사된 CSV 파일: {total_copied}개")
    print(f"  총 스킵된 CSV 파일: {total_skipped}개")


def main():
    parser = argparse.ArgumentParser(
        description="파단 위치가 포함된 CSV 파일만 선택하여 복사"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="4. edit_pole_data/break",
        help="입력 디렉토리 (기본값: 4. edit_pole_data/break)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="5. select_pole_data",
        help="출력 디렉토리 (기본값: 5. select_pole_data)",
    )
    
    args = parser.parse_args()
    
    # 현재 디렉토리 기준으로 경로 변환
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(current_dir, args.input_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(current_dir, args.output_dir)
    
    process_all_poles(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
