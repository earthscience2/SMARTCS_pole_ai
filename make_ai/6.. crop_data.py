#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파단 위치 정보를 참고하여 processed CSV 파일에서 파단 부분만 크롭하는 스크립트.
정상 데이터도 중간 높이 기준으로 동일한 범위로 크롭.

대상:
- 파단 데이터:
  - 입력: make_ai/4. edit_pole_data/break/ 프로젝트/전주ID/*_processed.csv
  - break_info: make_ai/4. edit_pole_data/break/ 프로젝트/전주ID/*_break_info.json
  - 출력: make_ai/5. crop_data/break/ 프로젝트/전주ID/*_break_crop.csv
- 정상 데이터:
  - 입력: make_ai/4. edit_pole_data/normal/ 프로젝트/전주ID/*_processed.csv
  - 출력: make_ai/5. crop_data/normal/ 프로젝트/전주ID/*_normal_crop.csv

동작:
1. 파단 데이터: break_info.json에서 breakheight 확인 후 ±0.15m 범위 크롭
2. 정상 데이터: 각 파일의 중간 높이 기준 ±0.15m 범위 크롭
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_break_info(break_info_path: str):
    """
    break_info.json에서 breakheight와 breakdegree 읽기.
    
    Returns:
        tuple: (breakheight, breakdegree) 또는 (None, None) (데이터가 없는 경우)
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


def check_file_contains_break_range(
    csv_path: str, 
    break_height: float, 
    break_degree: Optional[float],
    height_margin: float = 0.15
) -> bool:
    """
    CSV 파일이 파단 높이 및 각도 범위를 포함하는지 확인.
    
    Args:
        csv_path: CSV 파일 경로
        break_height: 파단 높이
        break_degree: 파단 각도 (None이면 각도 확인 안 함)
        height_margin: 파단 높이 기준 ± margin (m)
    
    Returns:
        bool: 파단 범위를 포함하면 True, 아니면 False
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "height" not in df.columns:
            return False
        
        # 높이 범위 확인
        h_min = break_height - height_margin
        h_max = break_height + height_margin
        
        file_h_min = df["height"].min()
        file_h_max = df["height"].max()
        
        # 높이 범위가 겹치는지 확인
        height_overlap = not (file_h_max < h_min or file_h_min > h_max)
        
        if not height_overlap:
            return False
        
        # 각도 확인 (break_degree가 있는 경우)
        if break_degree is not None:
            if "degree" not in df.columns:
                return False
            
            file_d_min = df["degree"].min()
            file_d_max = df["degree"].max()
            
            # 각도 범위 확인 (0~360도 wrap-around 고려)
            def is_degree_in_range(deg, d_min, d_max):
                # 범위가 0을 넘어가는 경우 (예: 350~10도)
                if d_min > d_max:
                    return deg >= d_min or deg <= d_max
                else:
                    return d_min <= deg <= d_max
            
            # break_degree가 파일의 각도 범위 안에 있는지 확인
            degree_contained = is_degree_in_range(break_degree, file_d_min, file_d_max)
            
            return degree_contained
        else:
            # break_degree가 없으면 높이만 확인
            return True
            
    except Exception as e:
        print(f"  경고: 파일 확인 중 오류 ({csv_path}): {e}")
        return False


def crop_break_region(
    processed_csv_path: str,
    break_height: float,
    height_margin: float = 0.15,
    output_dir: str = None
):
    """
    processed CSV에서 파단 위치 주변(높이만)을 크롭.

    Args:
        processed_csv_path: processed CSV 파일 경로
        break_height: 파단 높이
        height_margin: breakheight 기준 ± margin (m)
        output_dir: 출력 디렉토리 경로 (None이면 원본 CSV와 같은 디렉토리)
    
    Returns:
        str or None: 저장된 파일 경로 또는 None (크롭 실패 시)
    """
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {processed_csv_path}")

    # CSV 로드
    df = pd.read_csv(processed_csv_path)
    if df.empty:
        print(f"  경고: CSV 데이터가 비어 있습니다: {processed_csv_path}")
        return None

    required_cols = {"height", "degree", "x_value", "y_value", "z_value", "devicetype"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"  경고: 필요한 컬럼이 없습니다 {missing_cols}: {processed_csv_path}")
        return None

    # 높이 범위 계산
    h_min = break_height - height_margin
    h_max = break_height + height_margin

    # 실제 데이터 범위에 맞게 클램프
    data_h_min = df["height"].min()
    data_h_max = df["height"].max()
    h_min = max(h_min, data_h_min)
    h_max = min(h_max, data_h_max)

    # 높이 필터링만 수행 (각도는 모든 각도 포함)
    height_mask = (df["height"] >= h_min) & (df["height"] <= h_max)

    cropped = df[height_mask].copy()
    cropped = cropped.sort_values(["height", "degree"]).reset_index(drop=True)

    if cropped.empty:
        print(f"  경고: 크롭 결과가 비어 있습니다: {processed_csv_path}")
        return None

    # 저장 경로 결정
    csv_path = Path(processed_csv_path)
    if output_dir is not None:
        # 지정된 출력 디렉토리 사용
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{csv_path.stem}_break_crop.csv"
    else:
        # 원본 CSV와 같은 디렉토리 (기본 동작)
        output_path = csv_path.with_name(f"{csv_path.stem}_break_crop.csv")

    cropped.to_csv(output_path, index=False)
    return str(output_path)


def crop_normal_region(
    processed_csv_path: str,
    height_margin: float = 0.15,
    output_dir: str = None
):
    """
    정상 데이터 processed CSV에서 중간 높이 기준으로 크롭.
    
    Args:
        processed_csv_path: processed CSV 파일 경로
        height_margin: 중간 높이 기준 ± margin (m)
        output_dir: 출력 디렉토리 경로 (None이면 원본 CSV와 같은 디렉토리)
    
    Returns:
        str or None: 저장된 파일 경로 또는 None (크롭 실패 시)
    """
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {processed_csv_path}")
    
    # CSV 로드
    df = pd.read_csv(processed_csv_path)
    if df.empty:
        print(f"  경고: CSV 데이터가 비어 있습니다: {processed_csv_path}")
        return None
    
    required_cols = {"height", "degree", "x_value", "y_value", "z_value", "devicetype"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"  경고: 필요한 컬럼이 없습니다 {missing_cols}: {processed_csv_path}")
        return None
    
    # 필터링: 높이가 2m 이상이거나 각도가 360도 이상이면 크롭하지 않음
    data_h_max = df["height"].max()
    data_d_max = df["degree"].max()
    
    if data_h_max >= 2.0 or data_d_max >= 360.0:
        print(f"  스킵: 높이 또는 각도 범위 초과 (높이 최대: {data_h_max:.3f}m, 각도 최대: {data_d_max:.1f}°): {processed_csv_path}")
        return None
    
    # 중간 높이 계산
    data_h_min = df["height"].min()
    center_height = (data_h_min + data_h_max) / 2.0
    
    # 높이 범위 계산
    h_min = center_height - height_margin
    h_max = center_height + height_margin
    
    # 실제 데이터 범위에 맞게 클램프
    h_min = max(h_min, data_h_min)
    h_max = min(h_max, data_h_max)
    
    # 높이 필터링만 수행 (각도는 모든 각도 포함)
    height_mask = (df["height"] >= h_min) & (df["height"] <= h_max)
    
    cropped = df[height_mask].copy()
    cropped = cropped.sort_values(["height", "degree"]).reset_index(drop=True)
    
    if cropped.empty:
        print(f"  경고: 크롭 결과가 비어 있습니다: {processed_csv_path}")
        return None
    
    # 저장 경로 결정
    csv_path = Path(processed_csv_path)
    if output_dir is not None:
        # 지정된 출력 디렉토리 사용
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{csv_path.stem}_normal_crop.csv"
    else:
        # 원본 CSV와 같은 디렉토리 (기본 동작)
        output_path = csv_path.with_name(f"{csv_path.stem}_normal_crop.csv")
    
    cropped.to_csv(output_path, index=False)
    return str(output_path)


def process_all_processed_files(
    input_base_dir: str = "4. edit_pole_data/break",
    output_base_dir: str = "5. crop_data/break",
    height_margin: float = 0.15,
):
    """
    edit_pole_data/break 디렉토리 아래의 모든 processed CSV 파일을 찾아서
    해당하는 break_info.json을 참고하여 파단 영역을 크롭.
    
    Args:
        input_base_dir: processed 파일이 있는 기본 디렉토리 (상대 경로)
        output_base_dir: 크롭된 파일을 저장할 기본 디렉토리 (상대 경로)
        height_margin: breakheight 기준 ± margin (m)
    """
    # 현재 스크립트 디렉토리를 기준으로 경로 생성
    input_path = Path(current_dir) / input_base_dir
    output_path = Path(current_dir) / output_base_dir
    
    if not input_path.exists():
        print(f"오류: 입력 디렉토리를 찾을 수 없습니다: {input_base_dir}")
        return
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다: {input_base_dir}")
        return
    
    total_processed = 0
    total_success = 0
    total_skipped = 0
    
    print(f"=" * 80)
    print(f"전체 processed 파일 크롭 시작")
    print(f"입력 디렉토리: {input_base_dir}")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"크롭 범위: 높이 ±{height_margin}m")
    print(f"=" * 80)
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        for pole_dir in pole_dirs:
            poleid = pole_dir.name
            
            # break_info.json 파일 찾기
            break_info_json = pole_dir / f"{poleid}_break_info.json"
            
            if not break_info_json.exists():
                print(f"  스킵 [{poleid}]: break_info.json 없음")
                total_skipped += 1
                continue
            
            # break_info.json에서 breakheight와 breakdegree 읽기
            break_height, break_degree = load_break_info(str(break_info_json))
            
            if break_height is None:
                print(f"  스킵 [{poleid}]: breakheight 정보가 없음")
                total_skipped += 1
                continue
            
            if break_degree is not None:
                print(f"  [{poleid}] 파단 높이: {break_height:.3f}m, 각도: {break_degree:.1f}°")
            else:
                print(f"  [{poleid}] 파단 높이: {break_height:.3f}m (각도 정보 없음)")
            
            # break_info.json에서 breakstate 확인 (B인 경우만 처리)
            try:
                with open(break_info_json, "r", encoding="utf-8") as f:
                    break_info = json.load(f)
                
                breakstate = break_info.get("breakstate")
                if breakstate != "B":
                    print(f"  스킵 [{poleid}]: breakstate가 'B'가 아님 (현재: {breakstate})")
                    total_skipped += 1
                    continue
                
            except Exception as e:
                print(f"  오류 [{poleid}]: break_info.json 읽기 실패 - {e}")
                total_skipped += 1
                continue
            
            # 모든 *_processed.csv 파일 찾기
            processed_files = list(pole_dir.glob("*_processed.csv"))
            
            if not processed_files:
                print(f"  스킵 [{poleid}]: processed CSV 파일이 없음")
                total_skipped += 1
                continue
            
            # 출력 디렉토리 생성 (프로젝트/전주ID 구조 유지)
            output_dir = output_path / project_name / poleid
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 각 processed 파일 확인 및 크롭
            files_cropped = 0
            for processed_csv in processed_files:
                csv_name = processed_csv.name
                
                # 파단 범위(높이 + 각도) 포함 여부 확인
                if not check_file_contains_break_range(str(processed_csv), break_height, break_degree, height_margin):
                    continue
                
                # 이미 크롭된 파일이 있으면 건너뛰기
                output_file = output_dir / f"{processed_csv.stem}_break_crop.csv"
                if output_file.exists():
                    continue
                
                # 크롭 수행
                total_processed += 1
                try:
                    result = crop_break_region(
                        processed_csv_path=str(processed_csv),
                        break_height=break_height,
                        height_margin=height_margin,
                        output_dir=str(output_dir),
                    )
                    if result is not None:
                        files_cropped += 1
                        total_success += 1
                        print(f"    크롭 완료: {csv_name} -> {output_file.name}")
                except Exception as e:
                    print(f"    오류 [{csv_name}]: {e}")
                    total_skipped += 1
            
            if files_cropped == 0:
                print(f"  스킵 [{poleid}]: 파단 범위를 포함하는 파일이 없거나 이미 처리됨")
    
    print(f"\n" + "=" * 80)
    print(f"전체 처리 완료")
    print(f"  처리 시도: {total_processed}개 파일")
    print(f"  성공: {total_success}개 파일")
    print(f"  스킵: {total_skipped}개 파일")
    print(f"=" * 80)


def process_all_normal_files(
    input_base_dir: str = "4. edit_pole_data/normal",
    output_base_dir: str = "5. crop_data/normal",
    height_margin: float = 0.15,
):
    """
    edit_pole_data/normal 디렉토리 아래의 모든 processed CSV 파일을 찾아서
    각 파일의 중간 높이 기준으로 크롭.
    
    Args:
        input_base_dir: processed 파일이 있는 기본 디렉토리 (상대 경로)
        output_base_dir: 크롭된 파일을 저장할 기본 디렉토리 (상대 경로)
        height_margin: 중간 높이 기준 ± margin (m)
    """
    # 현재 스크립트 디렉토리를 기준으로 경로 생성
    input_path = Path(current_dir) / input_base_dir
    output_path = Path(current_dir) / output_base_dir
    
    if not input_path.exists():
        print(f"오류: 입력 디렉토리를 찾을 수 없습니다: {input_base_dir}")
        return
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다: {input_base_dir}")
        return
    
    total_processed = 0
    total_success = 0
    total_skipped = 0
    
    print(f"=" * 80)
    print(f"정상 데이터 크롭 시작")
    print(f"입력 디렉토리: {input_base_dir}")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"크롭 범위: 중간 높이 ±{height_margin}m")
    print(f"=" * 80)
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        for pole_dir in pole_dirs:
            poleid = pole_dir.name
            
            # 모든 *_processed.csv 파일 찾기
            processed_files = list(pole_dir.glob("*_processed.csv"))
            
            if not processed_files:
                print(f"  스킵 [{poleid}]: processed CSV 파일이 없음")
                total_skipped += 1
                continue
            
            # 출력 디렉토리 생성 (프로젝트/전주ID 구조 유지)
            output_dir = output_path / project_name / poleid
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 각 processed 파일 크롭
            files_cropped = 0
            for processed_csv in processed_files:
                csv_name = processed_csv.name
                
                # 이미 크롭된 파일이 있으면 건너뛰기
                output_file = output_dir / f"{processed_csv.stem}_normal_crop.csv"
                if output_file.exists():
                    continue
                
                # 크롭 수행
                total_processed += 1
                try:
                    result = crop_normal_region(
                        processed_csv_path=str(processed_csv),
                        height_margin=height_margin,
                        output_dir=str(output_dir),
                    )
                    if result is not None:
                        files_cropped += 1
                        total_success += 1
                        print(f"    크롭 완료: {csv_name} -> {output_file.name}")
                except Exception as e:
                    print(f"    오류 [{csv_name}]: {e}")
                    total_skipped += 1
            
            if files_cropped == 0:
                print(f"  스킵 [{poleid}]: 이미 처리된 파일이거나 크롭 실패")
    
    print(f"\n" + "=" * 80)
    print(f"정상 데이터 처리 완료")
    print(f"  처리 시도: {total_processed}개 파일")
    print(f"  성공: {total_success}개 파일")
    print(f"  스킵: {total_skipped}개 파일")
    print(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="processed CSV 파일과 break_info.json을 이용해 파단 주변 영역(높이 ±0.15m)을 크롭하여 별도 CSV로 저장. 인자 없이 실행 시 모든 항목 자동 처리."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="4. edit_pole_data/break",
        help="processed 파일이 있는 기본 디렉토리 (상대 경로, 전체 처리용)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="5. crop_data/break",
        help="크롭된 파일을 저장할 기본 디렉토리 (상대 경로, 전체 처리용)",
    )
    parser.add_argument(
        "--height-margin",
        type=float,
        default=0.15,
        help="breakheight 기준 ± margin (m), 기본값: 0.15",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="단일 CSV 파일 경로 (단일 파일 처리용, --break-height와 함께 사용)",
    )
    parser.add_argument(
        "--break-height",
        type=float,
        default=None,
        help="파단 높이 (단일 파일 처리용, --csv-file과 함께 사용)",
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="정상 데이터 크롭 건너뛰기",
    )
    parser.add_argument(
        "--normal-input-dir",
        type=str,
        default="4. edit_pole_data/normal",
        help="정상 데이터 입력 디렉토리 (상대 경로)",
    )
    parser.add_argument(
        "--normal-output-dir",
        type=str,
        default="5. crop_data/normal",
        help="정상 데이터 출력 디렉토리 (상대 경로)",
    )

    args = parser.parse_args()

    # 단일 파일 처리 모드
    if args.csv_file is not None and args.break_height is not None:
        result = crop_break_region(
            processed_csv_path=args.csv_file,
            break_height=args.break_height,
            height_margin=args.height_margin,
            output_dir=None,  # 단일 파일 처리 시 원본과 같은 디렉토리에 저장
        )
        if result is None:
            print("크롭 실패했습니다.")
        else:
            print(f"크롭 완료: {result}")
    else:
        # 전체 처리 모드 (기본 동작)
        # 파단 데이터 처리
        process_all_processed_files(
            input_base_dir=args.input_dir,
            output_base_dir=args.output_dir,
            height_margin=args.height_margin,
        )
        
        # 정상 데이터 처리 (기본적으로 항상 실행, --skip-normal 옵션으로 건너뛸 수 있음)
        if not args.skip_normal:
            print("\n")
            process_all_normal_files(
                input_base_dir=args.normal_input_dir,
                output_base_dir=args.normal_output_dir,
                height_margin=args.height_margin,
            )

if __name__ == "__main__":
    main()



