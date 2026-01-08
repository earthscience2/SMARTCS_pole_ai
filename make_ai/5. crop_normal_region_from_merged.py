#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정상 전주의 머지된 OUT_merged.csv에서 랜덤 영역을 크롭하여 저장하는 스크립트.

파단 크롭과 동일한 크기(±20도, ±0.2m)로 랜덤하게 영역을 선택하여 크롭.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def crop_random_region(
    merged_csv_path: str,
    height_margin: float = 0.2,
    degree_margin: float = 20.0,
    output_dir: str = None,
    min_points: int = 10,
) -> Optional[str]:
    """
    머지된 CSV에서 랜덤 영역을 크롭.
    
    Args:
        merged_csv_path: 머지된 OUT_merged.csv 경로
        height_margin: 높이 범위 ± margin (m)
        degree_margin: 각도 범위 ± margin (deg)
        output_dir: 출력 디렉토리 경로
        min_points: 최소 데이터 포인트 수
    
    Returns:
        str or None: 저장된 파일 경로 또는 None
    """
    if not os.path.exists(merged_csv_path):
        return None
    
    try:
        # 머지 CSV 로드
        df = pd.read_csv(merged_csv_path)
        if df.empty:
            return None
        
        required_cols = {"height", "degree", "x_value", "y_value", "z_value"}
        if not required_cols.issubset(df.columns):
            return None
        
        # NaN 값 제거
        df = df.dropna(subset=list(required_cols))
        if len(df) < min_points:
            return None
        
        # 데이터 범위 확인
        height_min = df["height"].min()
        height_max = df["height"].max()
        degree_min = df["degree"].min()
        degree_max = df["degree"].max()
        
        # 랜덤 중심점 선택
        # 높이: 데이터 범위 내에서 선택 (margin 고려)
        center_height = random.uniform(
            height_min + height_margin,
            height_max - height_margin
        )
        
        # 각도: 데이터 범위 내에서 선택 (margin 고려, 360도 wrap-around 고려)
        # 각도 범위가 충분히 큰 경우
        if degree_max - degree_min >= 2 * degree_margin:
            center_degree = random.uniform(
                degree_min + degree_margin,
                degree_max - degree_margin
            )
        else:
            # 범위가 작으면 중앙값 사용
            center_degree = (degree_min + degree_max) / 2.0
        
        # 크롭 범위 계산
        h_min = center_height - height_margin
        h_max = center_height + height_margin
        d_min = center_degree - degree_margin
        d_max = center_degree + degree_margin
        
        # 실제 데이터 범위에 맞게 클램프
        h_min = max(h_min, height_min)
        h_max = min(h_max, height_max)
        
        # degree는 0~360 기준, wrap-around 고려
        def in_degree_range(deg_series: pd.Series, dmin: float, dmax: float) -> pd.Series:
            dmin_wrapped = dmin % 360.0
            dmax_wrapped = dmax % 360.0
            if dmin_wrapped <= dmax_wrapped:
                return (deg_series >= dmin_wrapped) & (deg_series <= dmax_wrapped)
            else:
                # 예: 350~10도 범위처럼 0을 넘어가는 경우
                return (deg_series >= dmin_wrapped) | (deg_series <= dmax_wrapped)
        
        height_mask = (df["height"] >= h_min) & (df["height"] <= h_max)
        degree_mask = in_degree_range(df["degree"], d_min, d_max)
        
        cropped = df[height_mask & degree_mask].copy()
        cropped = cropped.sort_values(["height", "degree"]).reset_index(drop=True)
        
        # 최소 포인트 수 확인
        if len(cropped) < min_points:
            return None
        
        # 저장 경로 결정
        merged_path = Path(merged_csv_path)
        if output_dir is not None:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = output_dir_path / f"{merged_path.stem}_normal_crop.csv"
        else:
            output_path = merged_path.with_name(f"{merged_path.stem}_normal_crop.csv")
        
        cropped.to_csv(output_path, index=False)
        return str(output_path)
        
    except Exception as e:
        print(f"  오류: {e}")
        return None


def process_all_normal_merged_files(
    merged_base_dir: str = "4. merge_pole_data/normal_merged",
    normal_base_dir: str = "3. raw_pole_data/normal",
    output_base_dir: str = "4. merge_pole_data/normal_crop",
    height_margin: float = 0.2,
    degree_margin: float = 20.0,
    samples_per_pole: int = 1,
    min_points: int = 10,
):
    """
    정상 전주의 모든 머지 파일에서 랜덤 영역을 크롭.
    
    Args:
        merged_base_dir: 머지 파일이 있는 기본 디렉토리
        normal_base_dir: 정상 전주 정보가 있는 기본 디렉토리 (선택사항)
        output_base_dir: 크롭된 파일을 저장할 기본 디렉토리
        height_margin: 높이 범위 ± margin (m)
        degree_margin: 각도 범위 ± margin (deg)
        samples_per_pole: 전주당 샘플 수
        min_points: 최소 데이터 포인트 수
    """
    merged_path = Path(current_dir) / merged_base_dir
    output_path = Path(current_dir) / output_base_dir
    
    if not merged_path.exists():
        print(f"오류: merged 디렉토리를 찾을 수 없습니다: {merged_base_dir}")
        return
    
    # 정상 전주 목록 확인 (normal_base_dir이 있는 경우)
    normal_poles = set()
    if normal_base_dir:
        normal_path = Path(current_dir) / normal_base_dir
        if normal_path.exists():
            for project_dir in normal_path.iterdir():
                if project_dir.is_dir():
                    for pole_dir in project_dir.iterdir():
                        if pole_dir.is_dir():
                            normal_poles.add((project_dir.name, pole_dir.name))
            print(f"정상 전주 정보: {len(normal_poles)}개 전주 발견")
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in merged_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다: {merged_base_dir}")
        return
    
    total_processed = 0
    total_success = 0
    total_skipped = 0
    
    print(f"=" * 80)
    print(f"정상 전주 랜덤 영역 크롭 시작")
    print(f"merged 디렉토리: {merged_base_dir}")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"크롭 범위: 높이 ±{height_margin}m, 각도 ±{degree_margin}도")
    print(f"전주당 샘플 수: {samples_per_pole}개")
    print(f"=" * 80)
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        for pole_dir in pole_dirs:
            poleid = pole_dir.name
            
            # 정상 전주 목록이 있고, 해당 전주가 정상이 아니면 스킵
            if normal_poles and (project_name, poleid) not in normal_poles:
                continue
            
            # OUT_merged.csv 파일 찾기
            merged_csv = pole_dir / f"{poleid}_OUT_merged.csv"
            if not merged_csv.exists():
                continue
            
            # 여러 샘플 생성
            success_count = 0
            for sample_idx in range(samples_per_pole):
                total_processed += 1
                try:
                    output_dir = output_path / project_name / poleid
                    result = crop_random_region(
                        merged_csv_path=str(merged_csv),
                        height_margin=height_margin,
                        degree_margin=degree_margin,
                        output_dir=str(output_dir),
                        min_points=min_points,
                    )
                    
                    if result is None:
                        total_skipped += 1
                    else:
                        total_success += 1
                        success_count += 1
                        
                except Exception as e:
                    print(f"  실패 [{poleid}] 샘플 {sample_idx+1}: {e}")
                    total_skipped += 1
            
            if success_count > 0:
                print(f"  성공 [{poleid}]: {success_count}개 샘플 생성")
    
    print(f"\n" + "=" * 80)
    print(f"전체 처리 완료")
    print(f"  처리 시도: {total_processed}개")
    print(f"  성공: {total_success}개")
    print(f"  스킵: {total_skipped}개")
    print(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="정상 전주의 머지된 OUT_merged.csv에서 랜덤 영역을 크롭"
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="4. merge_pole_data/normal_merged",
        help="merged 파일이 있는 기본 디렉토리",
    )
    parser.add_argument(
        "--normal-dir",
        type=str,
        default="3. raw_pole_data/normal",
        help="정상 전주 정보가 있는 기본 디렉토리 (선택사항, 생략 시 모든 전주 처리)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="4. merge_pole_data/normal_crop",
        help="크롭된 파일을 저장할 기본 디렉토리",
    )
    parser.add_argument(
        "--height-margin",
        type=float,
        default=0.2,
        help="높이 범위 ± margin (m), 기본값: 0.2",
    )
    parser.add_argument(
        "--degree-margin",
        type=float,
        default=20.0,
        help="각도 범위 ± margin (deg), 기본값: 20도",
    )
    parser.add_argument(
        "--samples-per-pole",
        type=int,
        default=1,
        help="전주당 생성할 샘플 수, 기본값: 1",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="최소 데이터 포인트 수, 기본값: 10",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("정상 전주 랜덤 영역 크롭 시작")
    print("=" * 80)
    
    process_all_normal_merged_files(
        merged_base_dir=args.merged_dir,
        normal_base_dir=args.normal_dir,
        output_base_dir=args.output_dir,
        height_margin=args.height_margin,
        degree_margin=args.degree_margin,
        samples_per_pole=args.samples_per_pole,
        min_points=args.min_points,
    )
    
    print("\n" + "=" * 80)
    print("정상 데이터 크롭 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()

