#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ai/5. select_pole_data 디렉토리의 모든 *_processed.csv 파일에 대해 2차원 그래프 생성

각 CSV 파일에 대해:
- 높이와 각도에 따른 x, y, z 값 등고선 그래프 (3개)
- 각 파일명에 대응하는 PNG 파일로 저장
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def plot_csv_2d(csv_path: str, output_path: Optional[str] = None, break_info_path: Optional[str] = None):
    """
    CSV 파일을 읽어서 높이와 각도에 따른 x, y, z 값을 등고선 형태로 표시
    
    Args:
        csv_path: 입력 CSV 파일 경로
        output_path: 출력 PNG 파일 경로 (None이면 CSV와 같은 디렉토리에 저장)
        break_info_path: break_info.json 파일 경로 (선택사항, 파단 위치 표시용)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"  경고: {os.path.basename(csv_path)} - 데이터가 비어 있습니다.")
            return False
        
        # 필요한 컬럼 확인
        required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  경고: {os.path.basename(csv_path)} - 필요한 컬럼이 없습니다: {missing_cols}")
            return False
        
        # 파단 정보 읽기 (있는 경우)
        break_height = None
        break_degree = None
        if break_info_path and os.path.exists(break_info_path):
            try:
                import json
                with open(break_info_path, 'r', encoding='utf-8') as f:
                    break_info = json.load(f)
                break_height = break_info.get('breakheight')
                break_degree = break_info.get('breakdegree')
                if break_height is not None:
                    break_height = float(break_height)
                if break_degree is not None:
                    break_degree = float(break_degree)
            except Exception as e:
                pass  # 파단 정보가 없어도 계속 진행
        
        # 데이터 정렬 (높이, 각도 순)
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
        
        # 높이와 각도의 고유값 추출
        heights = sorted(df['height'].unique())
        degrees = sorted(df['degree'].unique())
        
        # 그리드 생성 (contourf는 (X, Y, Z) 형태를 받음)
        # X: degree 좌표 그리드, Y: height 좌표 그리드
        degree_grid, height_grid = np.meshgrid(degrees, heights)
        
        # x, y, z 값을 그리드 형태로 변환 (shape: (len(heights), len(degrees)))
        x_grid = np.full((len(heights), len(degrees)), np.nan)
        y_grid = np.full((len(heights), len(degrees)), np.nan)
        z_grid = np.full((len(heights), len(degrees)), np.nan)
        
        # 데이터를 그리드에 매핑
        for _, row in df.iterrows():
            h = row['height']
            d = row['degree']
            
            # 그리드 인덱스 찾기
            h_idx = heights.index(h) if h in heights else None
            d_idx = degrees.index(d) if d in degrees else None
            
            if h_idx is not None and d_idx is not None:
                x_grid[h_idx, d_idx] = row['x_value'] if pd.notna(row['x_value']) else np.nan
                y_grid[h_idx, d_idx] = row['y_value'] if pd.notna(row['y_value']) else np.nan
                z_grid[h_idx, d_idx] = row['z_value'] if pd.notna(row['z_value']) else np.nan
        
        # 각 축별 값 범위 계산 (NaN 제외)
        x_min, x_max = np.nanmin(x_grid), np.nanmax(x_grid)
        y_min, y_max = np.nanmin(y_grid), np.nanmax(y_grid)
        z_min, z_max = np.nanmin(z_grid), np.nanmax(z_grid)
        
        # 등고선 레벨 설정 (높이차가 잘 보이도록 세밀하게)
        n_levels = 50
        x_levels = np.linspace(x_min, x_max, n_levels)
        y_levels = np.linspace(y_min, y_max, n_levels)
        z_levels = np.linspace(z_min, z_max, n_levels)
        
        # 등고선 라인 레벨
        n_line_levels = 15
        x_line_levels = np.linspace(x_min, x_max, n_line_levels)
        y_line_levels = np.linspace(y_min, y_max, n_line_levels)
        z_line_levels = np.linspace(z_min, z_max, n_line_levels)
        
        # 그래프 생성 (3개의 서브플롯: x, y, z)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(os.path.basename(csv_path), fontsize=14, fontweight='bold')
        
        # X 값 등고선
        ax_x = axes[0]
        contour_x = ax_x.contourf(degree_grid, height_grid, x_grid, levels=x_levels, 
                                   cmap='hot', extend='both', alpha=0.9)
        contour_lines_x = ax_x.contour(degree_grid, height_grid, x_grid, levels=x_line_levels,
                                        colors='black', linewidths=0.5, alpha=0.5, linestyles='solid')
        ax_x.set_xlabel('Degree (°)', fontsize=11)
        ax_x.set_ylabel('Height (m)', fontsize=11)
        ax_x.set_title('X Value Contour', fontsize=12, fontweight='bold')
        ax_x.set_ylim(min(heights), max(heights))
        ax_x.set_xlim(min(degrees), max(degrees))
        plt.colorbar(contour_x, ax=ax_x, label='X Value (m)', shrink=0.8)
        
        # 파단 위치 표시 (X)
        if break_height is not None and break_degree is not None:
            ax_x.plot(break_degree, break_height, 'r*', markersize=15, 
                     markeredgecolor='white', markeredgewidth=2, label='Break Point', zorder=10)
            ax_x.legend(loc='upper right', fontsize=9)
        elif break_height is not None:
            ax_x.axhline(y=break_height, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break Height')
            ax_x.legend(loc='upper right', fontsize=9)
        
        # Y 값 등고선
        ax_y = axes[1]
        contour_y = ax_y.contourf(degree_grid, height_grid, y_grid, levels=y_levels, 
                                   cmap='hot', extend='both', alpha=0.9)
        contour_lines_y = ax_y.contour(degree_grid, height_grid, y_grid, levels=y_line_levels,
                                        colors='black', linewidths=0.5, alpha=0.5, linestyles='solid')
        ax_y.set_xlabel('Degree (°)', fontsize=11)
        ax_y.set_ylabel('Height (m)', fontsize=11)
        ax_y.set_title('Y Value Contour', fontsize=12, fontweight='bold')
        ax_y.set_ylim(min(heights), max(heights))
        ax_y.set_xlim(min(degrees), max(degrees))
        plt.colorbar(contour_y, ax=ax_y, label='Y Value (m)', shrink=0.8)
        
        # 파단 위치 표시 (Y)
        if break_height is not None and break_degree is not None:
            ax_y.plot(break_degree, break_height, 'r*', markersize=15, 
                     markeredgecolor='white', markeredgewidth=2, label='Break Point', zorder=10)
            ax_y.legend(loc='upper right', fontsize=9)
        elif break_height is not None:
            ax_y.axhline(y=break_height, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break Height')
            ax_y.legend(loc='upper right', fontsize=9)
        
        # Z 값 등고선
        ax_z = axes[2]
        contour_z = ax_z.contourf(degree_grid, height_grid, z_grid, levels=z_levels, 
                                   cmap='hot', extend='both', alpha=0.9)
        contour_lines_z = ax_z.contour(degree_grid, height_grid, z_grid, levels=z_line_levels,
                                        colors='black', linewidths=0.5, alpha=0.5, linestyles='solid')
        ax_z.set_xlabel('Degree (°)', fontsize=11)
        ax_z.set_ylabel('Height (m)', fontsize=11)
        ax_z.set_title('Z Value Contour', fontsize=12, fontweight='bold')
        ax_z.set_ylim(min(heights), max(heights))
        ax_z.set_xlim(min(degrees), max(degrees))
        plt.colorbar(contour_z, ax=ax_z, label='Z Value (m)', shrink=0.8)
        
        # 파단 위치 표시 (Z)
        if break_height is not None and break_degree is not None:
            ax_z.plot(break_degree, break_height, 'r*', markersize=15, 
                     markeredgecolor='white', markeredgewidth=2, label='Break Point', zorder=10)
            ax_z.legend(loc='upper right', fontsize=9)
        elif break_height is not None:
            ax_z.axhline(y=break_height, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break Height')
            ax_z.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # 출력 경로 결정
        if output_path is None:
            csv_path_obj = Path(csv_path)
            output_path = csv_path_obj.parent / f"{csv_path_obj.stem}_2d_plot.png"
        else:
            output_path = Path(output_path)
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  오류: {os.path.basename(csv_path)} 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_csv_files(input_dir: str, output_dir: Optional[str] = None):
    """
    입력 디렉토리의 모든 *_processed.csv 파일에 대해 그래프 생성
    
    Args:
        input_dir: 입력 디렉토리 (프로젝트/전주ID 구조)
        output_dir: 출력 디렉토리 (None이면 입력 디렉토리와 같은 위치에 저장)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return
    
    # 모든 *_processed.csv 파일 찾기
    csv_files = list(input_path.rglob("*_processed.csv"))
    
    if not csv_files:
        print(f"경고: {input_dir}에서 *_processed.csv 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    print(f"그래프 생성 시작...\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, csv_file in enumerate(csv_files, 1):
        # break_info.json 파일 찾기
        csv_dir = csv_file.parent
        poleid = csv_dir.name
        
        # break_info.json 파일 찾기 (여러 패턴 시도)
        break_info_path = None
        for pattern in [f"{poleid}_break_info.json", "break_info.json"]:
            candidate = csv_dir / pattern
            if candidate.exists():
                break_info_path = str(candidate)
                break
        
        # 출력 경로 결정
        if output_dir is None:
            output_file = csv_file.parent / f"{csv_file.stem}_2d_plot.png"
        else:
            # 입력 디렉토리 기준 상대 경로 유지
            rel_path = csv_file.relative_to(input_path)
            output_file = Path(output_dir) / rel_path.parent / f"{csv_file.stem}_2d_plot.png"
        
        print(f"[{idx}/{len(csv_files)}] 처리 중: {csv_file.name}")
        
        if plot_csv_2d(str(csv_file), str(output_file), break_info_path):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n완료!")
    print(f"  성공: {success_count}개")
    print(f"  실패: {fail_count}개")


def main():
    parser = argparse.ArgumentParser(
        description="processed CSV 파일들에 대해 2차원 그래프 생성"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="5. select_pole_data",
        help="입력 디렉토리 (기본값: 5. select_pole_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본값: 입력 파일과 같은 디렉토리)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="단일 CSV 파일 처리 (선택사항)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="단일 파일 처리 시 출력 파일 경로 (선택사항)",
    )
    
    args = parser.parse_args()
    
    # 현재 디렉토리 기준으로 경로 변환
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(current_dir, args.input_dir)
    if args.output_dir and not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(current_dir, args.output_dir)
    if args.csv_file and not os.path.isabs(args.csv_file):
        args.csv_file = os.path.join(current_dir, args.csv_file)
    if args.output_file and not os.path.isabs(args.output_file):
        args.output_file = os.path.join(current_dir, args.output_file)
    
    # 단일 파일 처리
    if args.csv_file:
        csv_dir = Path(args.csv_file).parent
        poleid = csv_dir.name
        
        # break_info.json 파일 찾기
        break_info_path = None
        for pattern in [f"{poleid}_break_info.json", "break_info.json"]:
            candidate = csv_dir / pattern
            if candidate.exists():
                break_info_path = str(candidate)
                break
        
        plot_csv_2d(args.csv_file, args.output_file, break_info_path)
    else:
        # 전체 디렉토리 처리
        process_all_csv_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
