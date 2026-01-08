#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
머지된 OUT_merged.csv와 break_info.json을 참고해서
파단 위치 주변(±20도, ±0.2m)을 크롭하여 별도 CSV로 저장하는 스크립트.

대상 예시:
- 머지 파일:
  make_ai/raw_pole_data/merged/강원동해-202209/0621R481/0621R481_OUT_merged.csv
- break_info:
  make_ai/raw_pole_data/break/강원동해-202209/0621R481/0621R481_break_info.json
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_break_info(break_info_path: str):
    """
    break_info.json에서 breakheight, breakdegree 읽기.
    
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

    if bh is None or bd is None:
        return None, None

    try:
        return float(bh), float(bd)
    except (ValueError, TypeError):
        return None, None


def crop_break_region(
    merged_csv_path: str,
    break_info_path: str,
    height_margin: float = 0.2,
    degree_margin: float = 20.0,
    output_dir: str = None,
):
    """
    머지된 CSV에서 파단 위치 주변을 크롭.

    Args:
        merged_csv_path: 머지된 OUT_merged.csv 경로
        break_info_path: break_info.json 경로
        height_margin: breakheight 기준 ± margin (m)
        degree_margin: breakdegree 기준 ± margin (deg)
        output_dir: 출력 디렉토리 경로 (None이면 원본 CSV와 같은 디렉토리)
    
    Returns:
        str or None: 저장된 파일 경로 또는 None (파단 위치 데이터가 없는 경우)
    """
    if not os.path.exists(merged_csv_path):
        raise FileNotFoundError(f"머지 CSV 파일을 찾을 수 없습니다: {merged_csv_path}")

    # 파단 위치 정보 로드
    break_height, break_degree = load_break_info(break_info_path)
    
    # 파단 위치 데이터가 없는 경우 건너뛰기
    if break_height is None or break_degree is None:
        print(f"  스킵: 파단 위치 데이터가 없습니다 (breakheight={break_height}, breakdegree={break_degree})")
        return None
    
    print(f"파단 위치: height={break_height:.3f} m, degree={break_degree:.1f}°")

    # 머지 CSV 로드
    df = pd.read_csv(merged_csv_path)
    if df.empty:
        raise ValueError(f"머지 CSV 데이터가 비어 있습니다: {merged_csv_path}")

    required_cols = {"height", "degree"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"머지 CSV에 필요한 컬럼 {required_cols} 이(가) 없습니다: {merged_csv_path}")

    # 높이/각도 범위 계산
    h_min = break_height - height_margin
    h_max = break_height + height_margin

    # 실제 데이터 범위에 맞게 클램프
    data_h_min = df["height"].min()
    data_h_max = df["height"].max()
    h_min = max(h_min, data_h_min)
    h_max = min(h_max, data_h_max)

    d_min = break_degree - degree_margin
    d_max = break_degree + degree_margin

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

    print(
        f"크롭 범위: height [{h_min:.3f}, {h_max:.3f}] m, "
        f"degree [{d_min:.1f}, {d_max:.1f}] (wrap-around 고려)"
    )
    print(f"크롭된 데이터 행 수: {len(cropped)}")

    if cropped.empty:
        print("경고: 크롭 결과가 비어 있습니다. 파일을 저장하지 않습니다.")
        return None

    # 저장 경로 결정
    merged_path = Path(merged_csv_path)
    if output_dir is not None:
        # 지정된 출력 디렉토리 사용
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{merged_path.stem}_break_crop.csv"
    else:
        # 원본 CSV와 같은 디렉토리 (기본 동작)
        output_path = merged_path.with_name(f"{merged_path.stem}_break_crop.csv")

    cropped.to_csv(output_path, index=False)
    print(f"크롭된 데이터 저장 완료: {output_path}")

    return str(output_path)


def process_all_merged_files(
    merged_base_dir: str = "4. merge_pole_data/break_merged",
    break_base_dir: str = "3. raw_pole_data/break",
    output_base_dir: str = "4. merge_pole_data/break_crop",
    height_margin: float = 0.2,
    degree_margin: float = 20.0,
):
    """
    merged 디렉토리 아래의 모든 OUT_merged.csv 파일을 찾아서
    해당하는 break_info.json이 있으면 파단 영역을 크롭.
    
    Args:
        merged_base_dir: merged 파일이 있는 기본 디렉토리 (상대 경로)
        break_base_dir: break_info.json이 있는 기본 디렉토리 (상대 경로)
        output_base_dir: 크롭된 파일을 저장할 기본 디렉토리 (상대 경로)
        height_margin: breakheight 기준 ± margin (m)
        degree_margin: breakdegree 기준 ± margin (deg)
    """
    # 현재 스크립트 디렉토리를 기준으로 경로 생성
    merged_path = Path(current_dir) / merged_base_dir
    break_path = Path(current_dir) / break_base_dir
    output_path = Path(current_dir) / output_base_dir
    
    if not merged_path.exists():
        print(f"오류: merged 디렉토리를 찾을 수 없습니다: {merged_base_dir}")
        return
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in merged_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다: {merged_base_dir}")
        return
    
    total_processed = 0
    total_success = 0
    total_skipped = 0
    
    print(f"=" * 80)
    print(f"전체 머지 파일 크롭 시작")
    print(f"merged 디렉토리: {merged_base_dir}")
    print(f"break 디렉토리: {break_base_dir}")
    print(f"출력 디렉토리: {output_base_dir}")
    print(f"크롭 범위: 높이 ±{height_margin}m, 각도 ±{degree_margin}도")
    print(f"=" * 80)
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        for pole_dir in pole_dirs:
            poleid = pole_dir.name
            
            # OUT_merged.csv 파일 찾기
            merged_csv = pole_dir / f"{poleid}_OUT_merged.csv"
            if not merged_csv.exists():
                continue
            
            # break_info.json 파일 찾기 (같은 경로 구조)
            break_info_json = break_path / project_name / poleid / f"{poleid}_break_info.json"
            
            if not break_info_json.exists():
                print(f"  스킵 [{poleid}]: break_info.json 없음")
                total_skipped += 1
                continue
            
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
            
            # 크롭 수행
            total_processed += 1
            try:
                print(f"  처리 중 [{poleid}]...")
                # 출력 디렉토리 생성 (프로젝트/전주ID 구조 유지)
                output_dir = output_path / project_name / poleid
                result = crop_break_region(
                    merged_csv_path=str(merged_csv),
                    break_info_path=str(break_info_json),
                    height_margin=height_margin,
                    degree_margin=degree_margin,
                    output_dir=str(output_dir),
                )
                if result is None:
                    # 파단 위치 데이터가 없어서 건너뛴 경우
                    total_skipped += 1
                else:
                    total_success += 1
            except Exception as e:
                print(f"  실패 [{poleid}]: {e}")
                total_skipped += 1
    
    print(f"\n" + "=" * 80)
    print(f"전체 처리 완료")
    print(f"  처리 시도: {total_processed}개")
    print(f"  성공: {total_success}개")
    print(f"  스킵: {total_skipped}개")
    print(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="머지된 OUT_merged.csv와 break_info.json을 이용해 파단 주변 영역을 크롭하여 별도 CSV로 저장. 인자 없이 실행 시 모든 항목 자동 처리."
    )
    parser.add_argument(
        "--merged",
        type=str,
        default=None,
        help="머지된 OUT_merged.csv 경로 (단일 파일 처리용, 생략 시 전체 처리)",
    )
    parser.add_argument(
        "--break-info",
        type=str,
        default=None,
        help="break_info.json 경로 (단일 파일 처리용)",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="4. merge_pole_data/break_merged",
        help="merged 파일이 있는 기본 디렉토리 (상대 경로, 전체 처리용)",
    )
    parser.add_argument(
        "--break-dir",
        type=str,
        default="3. raw_pole_data/break",
        help="break_info.json이 있는 기본 디렉토리 (상대 경로, 전체 처리용)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="4. merge_pole_data/break_crop",
        help="크롭된 파일을 저장할 기본 디렉토리 (상대 경로, 전체 처리용)",
    )
    parser.add_argument(
        "--height-margin",
        type=float,
        default=0.2,
        help="breakheight 기준 ± margin (m), 기본값: 0.2",
    )
    parser.add_argument(
        "--degree-margin",
        type=float,
        default=20.0,
        help="breakdegree 기준 ± margin (deg), 기본값: 20도",
    )

    args = parser.parse_args()

    # 단일 파일 처리 모드
    if args.merged is not None and args.break_info is not None:
        result = crop_break_region(
            merged_csv_path=args.merged,
            break_info_path=args.break_info,
            height_margin=args.height_margin,
            degree_margin=args.degree_margin,
            output_dir=None,  # 단일 파일 처리 시 원본과 같은 디렉토리에 저장
        )
        if result is None:
            print("파단 위치 데이터가 없어서 크롭을 건너뛰었습니다.")
    else:
        # 전체 처리 모드 (기본 동작)
        process_all_merged_files(
            merged_base_dir=args.merged_dir,
            break_base_dir=args.break_dir,
            output_base_dir=args.output_dir,
            height_margin=args.height_margin,
            degree_margin=args.degree_margin,
        )


if __name__ == "__main__":
    main()


