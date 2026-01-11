#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
머지된 전주 데이터를 등고선(contour) 2D 그래프로 시각화
x, y, z 값을 각각 등고선으로 표시하고 하나의 화면에 배치
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import argparse
import os
import sys
from pathlib import Path

# Matplotlib 폰트 관련 경고 억제
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)

# 현재 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 기본 설정
# 직접 파일 경로를 지정하려면 아래 경로를 수정하세요
DEFAULT_CSV_FILE = os.path.join(
    current_dir, 
    "raw_pole_data/merged/강원동해-202209/0621R481/0621R481_OUT_merged.csv"
)

# 또는 절대 경로로 직접 지정:
# DEFAULT_CSV_FILE = "/Users/heegulee/Desktop/SMARTCS/make_ai/raw_pole_data/merged/강원동해-202209/0621R481/0621R481_OUT_merged.csv"

def plot_contour_2d(csv_file, output_file=None, break_info=None):
    """
    머지된 CSV 파일을 등고선 2D 그래프로 시각화
    
    Args:
        csv_file: 머지된 CSV 파일 경로
        output_file: 출력 이미지 파일 경로 (None이면 자동 생성)
        break_info: 파단 정보 딕셔너리 (breakheight, breakdegree 포함, 선택사항)
    """
    # CSV 파일 읽기
    if not os.path.exists(csv_file):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # 필수 컬럼 확인
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    for col in required_cols:
        if col not in df.columns:
            print(f"오류: 필수 컬럼 '{col}'이 없습니다.")
            return
    
    # 데이터 정렬 (높이, 각도 순) - 높이는 낮은 값부터 높은 값 순서로
    df = df.sort_values(['height', 'degree'])
    
    # 높이와 각도의 고유값 추출 (높이는 낮은 값부터 높은 값 순서로)
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
    
    # 각 축별 개별 범위 설정
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # 각 축별 등고선 레벨 설정 (더 세밀하게 - 급격한 변화를 잘 보이게)
    n_levels = 40  # 레벨 수 증가 (20 -> 40)
    x_levels = np.linspace(x_min, x_max, n_levels)
    y_levels = np.linspace(y_min, y_max, n_levels)
    z_levels = np.linspace(z_min, z_max, n_levels)
    
    # 등고선 라인 레벨 (더 적은 수로 명확한 경계 표시)
    n_line_levels = 15
    x_line_levels = np.linspace(x_min, x_max, n_line_levels)
    y_line_levels = np.linspace(y_min, y_max, n_line_levels)
    z_line_levels = np.linspace(z_min, z_max, n_line_levels)
    
    # 파단 정보 확인
    breakheight = None
    breakdegree = None
    if break_info is not None:
        bh = break_info.get('breakheight')
        bd = break_info.get('breakdegree')
        
        if bh is not None and not (isinstance(bh, float) and np.isnan(bh)):
            breakheight = float(bh)
        if bd is not None and not (isinstance(bd, float) and np.isnan(bd)):
            breakdegree = float(bd)
    
    # 그래프 생성 (3행 3열: 위는 등고선, 중간은 파단 높이에서의 각도별 값, 아래는 파단 각도에서의 높이별 값)
    if breakheight is not None and breakdegree is not None:
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Merged Pole Data - Contour Plot (X, Y, Z)', fontsize=16, fontweight='bold')
        contour_axes = axes[0]      # 1행: 등고선
        degree_line_axes = axes[1] # 2행: 각도별 값 그래프
        height_line_axes = axes[2]  # 3행: 높이별 값 그래프
    elif breakheight is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Merged Pole Data - Contour Plot (X, Y, Z)', fontsize=16, fontweight='bold')
        contour_axes = axes[0]      # 1행: 등고선
        degree_line_axes = axes[1]  # 2행: 각도별 값 그래프
        height_line_axes = None
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Merged Pole Data - Contour Plot (X, Y, Z)', fontsize=16, fontweight='bold')
        contour_axes = axes
        degree_line_axes = None
        height_line_axes = None
    
    # X 값 등고선
    ax1 = contour_axes[0]
    # 등고선 채우기 (더 대비가 강한 색상 맵 사용: 'hot' 또는 'Reds'의 변형)
    contour1 = ax1.contourf(degree_grid, height_grid, x_grid, levels=x_levels, 
                            cmap='hot', extend='both', alpha=0.9)
    # 등고선 라인 추가 (급격한 변화를 명확하게 표시)
    contour_lines1 = ax1.contour(degree_grid, height_grid, x_grid, levels=x_line_levels,
                                colors='black', linewidths=0.5, alpha=0.6, linestyles='solid')
    # 등고선 라인 레이블 표시 (선택적)
    ax1.clabel(contour_lines1, inline=True, fontsize=7, fmt='%1.3f', colors='white')
    ax1.set_xlabel('Degree (°)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('X Value', fontsize=14, fontweight='bold')
    ax1.set_ylim(min(heights), max(heights))  # 높이 축 범위 명시적 설정
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('X Value', fontsize=11)
    
    # Y 값 등고선
    ax2 = contour_axes[1]
    # 등고선 채우기
    contour2 = ax2.contourf(degree_grid, height_grid, y_grid, levels=y_levels,
                            cmap='viridis', extend='both', alpha=0.9)
    # 등고선 라인 추가
    contour_lines2 = ax2.contour(degree_grid, height_grid, y_grid, levels=y_line_levels,
                                 colors='white', linewidths=0.5, alpha=0.7, linestyles='solid')
    ax2.clabel(contour_lines2, inline=True, fontsize=7, fmt='%1.3f', colors='white')
    ax2.set_xlabel('Degree (°)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Y Value', fontsize=14, fontweight='bold')
    ax2.set_ylim(min(heights), max(heights))  # 높이 축 범위 명시적 설정
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Y Value', fontsize=11)
    
    # Z 값 등고선
    ax3 = contour_axes[2]
    # 등고선 채우기
    contour3 = ax3.contourf(degree_grid, height_grid, z_grid, levels=z_levels,
                            cmap='plasma', extend='both', alpha=0.9)
    # 등고선 라인 추가
    contour_lines3 = ax3.contour(degree_grid, height_grid, z_grid, levels=z_line_levels,
                                 colors='white', linewidths=0.5, alpha=0.7, linestyles='solid')
    ax3.clabel(contour_lines3, inline=True, fontsize=7, fmt='%1.3f', colors='white')
    ax3.set_xlabel('Degree (°)', fontsize=12)
    ax3.set_ylabel('Height (m)', fontsize=12)
    ax3.set_title('Z Value', fontsize=14, fontweight='bold')
    ax3.set_ylim(min(heights), max(heights))  # 높이 축 범위 명시적 설정
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar3 = plt.colorbar(contour3, ax=ax3)
    cbar3.set_label('Z Value', fontsize=11)
    
    # 파단 위치 표시 및 각도별 값 그래프 생성
    if breakheight is not None and breakdegree is not None:
        # 모든 등고선 서브플롯에 파단 위치 표시
        for ax in contour_axes:
            # 파단 위치에 빨간색 X 마커 표시
            ax.scatter(breakdegree, breakheight, color='red', marker='X', s=200, 
                      edgecolors='white', linewidths=2, zorder=10, label='Break Position')
            # 파단 위치에 텍스트 표시
            ax.annotate('Break', xy=(breakdegree, breakheight), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       zorder=11)
            ax.legend(loc='upper right', fontsize=9)
        
        # 파단 높이에서의 각도별 값 추출
        breakheight_rounded = round(breakheight, 1)  # 높이를 반올림하여 매칭
        
        # 파단 높이에 가장 가까운 높이 찾기
        closest_height = min(heights, key=lambda x: abs(x - breakheight_rounded))
        
        # 해당 높이에서의 데이터 추출
        height_data = df[df['height'] == closest_height].copy()
        height_data = height_data.sort_values('degree')
        
        # 각 축별로 각도별 값 그래프 그리기
        if not height_data.empty and degree_line_axes is not None:
            # X 값 그래프 (각도별)
            ax_x_degree = degree_line_axes[0]
            x_values = height_data['x_value'].values
            degrees_for_plot = height_data['degree'].values
            ax_x_degree.plot(degrees_for_plot, x_values, 'r-', linewidth=2, label='X Value', marker='o', markersize=3)
            if breakdegree is not None:
                # 파단 각도 위치에 수직선 표시
                ax_x_degree.axvline(x=breakdegree, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Degree ({breakdegree:.1f}°)')
                # 파단 각도에서의 값 표시
                x_at_break = height_data[height_data['degree'] == breakdegree]['x_value']
                if not x_at_break.empty:
                    ax_x_degree.scatter(breakdegree, x_at_break.iloc[0], color='red', s=150, 
                                       marker='X', zorder=10, edgecolors='white', linewidths=2)
            ax_x_degree.set_xlabel('Degree (°)', fontsize=12)
            ax_x_degree.set_ylabel('X Value', fontsize=12)
            ax_x_degree.set_title(f'X Value at Height {closest_height:.1f}m (Break: {breakheight:.1f}m)', 
                                 fontsize=12, fontweight='bold')
            ax_x_degree.grid(True, alpha=0.3)
            ax_x_degree.legend(fontsize=9)
            
            # Y 값 그래프 (각도별)
            ax_y_degree = degree_line_axes[1]
            y_values = height_data['y_value'].values
            ax_y_degree.plot(degrees_for_plot, y_values, 'g-', linewidth=2, label='Y Value', marker='o', markersize=3)
            if breakdegree is not None:
                ax_y_degree.axvline(x=breakdegree, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Degree ({breakdegree:.1f}°)')
                y_at_break = height_data[height_data['degree'] == breakdegree]['y_value']
                if not y_at_break.empty:
                    ax_y_degree.scatter(breakdegree, y_at_break.iloc[0], color='red', s=150, 
                                       marker='X', zorder=10, edgecolors='white', linewidths=2)
            ax_y_degree.set_xlabel('Degree (°)', fontsize=12)
            ax_y_degree.set_ylabel('Y Value', fontsize=12)
            ax_y_degree.set_title(f'Y Value at Height {closest_height:.1f}m (Break: {breakheight:.1f}m)', 
                                 fontsize=12, fontweight='bold')
            ax_y_degree.grid(True, alpha=0.3)
            ax_y_degree.legend(fontsize=9)
            
            # Z 값 그래프 (각도별)
            ax_z_degree = degree_line_axes[2]
            z_values = height_data['z_value'].values
            ax_z_degree.plot(degrees_for_plot, z_values, 'b-', linewidth=2, label='Z Value', marker='o', markersize=3)
            if breakdegree is not None:
                ax_z_degree.axvline(x=breakdegree, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Degree ({breakdegree:.1f}°)')
                z_at_break = height_data[height_data['degree'] == breakdegree]['z_value']
                if not z_at_break.empty:
                    ax_z_degree.scatter(breakdegree, z_at_break.iloc[0], color='red', s=150, 
                                       marker='X', zorder=10, edgecolors='white', linewidths=2)
            ax_z_degree.set_xlabel('Degree (°)', fontsize=12)
            ax_z_degree.set_ylabel('Z Value', fontsize=12)
            ax_z_degree.set_title(f'Z Value at Height {closest_height:.1f}m (Break: {breakheight:.1f}m)', 
                                 fontsize=12, fontweight='bold')
            ax_z_degree.grid(True, alpha=0.3)
            ax_z_degree.legend(fontsize=9)
        
        # 파단 각도에서의 높이별 값 추출
        if breakdegree is not None and height_line_axes is not None:
            # 파단 각도에 가장 가까운 각도 찾기
            closest_degree = min(degrees, key=lambda x: abs(x - breakdegree))
            
            # 해당 각도에서의 데이터 추출
            degree_data = df[df['degree'] == closest_degree].copy()
            degree_data = degree_data.sort_values('height')
            
            # 각 축별로 높이별 값 그래프 그리기
            if not degree_data.empty:
                # X 값 그래프 (높이별)
                ax_x_height = height_line_axes[0]
                x_values_height = degree_data['x_value'].values
                heights_for_plot = degree_data['height'].values
                ax_x_height.plot(heights_for_plot, x_values_height, 'r-', linewidth=2, label='X Value', marker='o', markersize=3)
                if breakheight is not None:
                    # 파단 높이 위치에 수평선 표시
                    ax_x_height.axvline(x=breakheight, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Height ({breakheight:.1f}m)')
                    # 파단 높이에서의 값 표시
                    x_at_break_height = degree_data[degree_data['height'] == closest_height]['x_value']
                    if not x_at_break_height.empty:
                        ax_x_height.scatter(closest_height, x_at_break_height.iloc[0], color='red', s=150, 
                                          marker='X', zorder=10, edgecolors='white', linewidths=2)
                ax_x_height.set_xlabel('Height (m)', fontsize=12)
                ax_x_height.set_ylabel('X Value', fontsize=12)
                ax_x_height.set_title(f'X Value at Degree {closest_degree:.1f}° (Break: {breakdegree:.1f}°)', 
                                     fontsize=12, fontweight='bold')
                ax_x_height.grid(True, alpha=0.3)
                ax_x_height.legend(fontsize=9)
                
                # Y 값 그래프 (높이별)
                ax_y_height = height_line_axes[1]
                y_values_height = degree_data['y_value'].values
                ax_y_height.plot(heights_for_plot, y_values_height, 'g-', linewidth=2, label='Y Value', marker='o', markersize=3)
                if breakheight is not None:
                    ax_y_height.axvline(x=breakheight, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Height ({breakheight:.1f}m)')
                    y_at_break_height = degree_data[degree_data['height'] == closest_height]['y_value']
                    if not y_at_break_height.empty:
                        ax_y_height.scatter(closest_height, y_at_break_height.iloc[0], color='red', s=150, 
                                           marker='X', zorder=10, edgecolors='white', linewidths=2)
                ax_y_height.set_xlabel('Height (m)', fontsize=12)
                ax_y_height.set_ylabel('Y Value', fontsize=12)
                ax_y_height.set_title(f'Y Value at Degree {closest_degree:.1f}° (Break: {breakdegree:.1f}°)', 
                                     fontsize=12, fontweight='bold')
                ax_y_height.grid(True, alpha=0.3)
                ax_y_height.legend(fontsize=9)
                
                # Z 값 그래프 (높이별)
                ax_z_height = height_line_axes[2]
                z_values_height = degree_data['z_value'].values
                ax_z_height.plot(heights_for_plot, z_values_height, 'b-', linewidth=2, label='Z Value', marker='o', markersize=3)
                if breakheight is not None:
                    ax_z_height.axvline(x=breakheight, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Break Height ({breakheight:.1f}m)')
                    z_at_break_height = degree_data[degree_data['height'] == closest_height]['z_value']
                    if not z_at_break_height.empty:
                        ax_z_height.scatter(closest_height, z_at_break_height.iloc[0], color='red', s=150, 
                                           marker='X', zorder=10, edgecolors='white', linewidths=2)
                ax_z_height.set_xlabel('Height (m)', fontsize=12)
                ax_z_height.set_ylabel('Z Value', fontsize=12)
                ax_z_height.set_title(f'Z Value at Degree {closest_degree:.1f}° (Break: {breakdegree:.1f}°)', 
                                     fontsize=12, fontweight='bold')
                ax_z_height.grid(True, alpha=0.3)
                ax_z_height.legend(fontsize=9)
    
    plt.tight_layout()
    
    # 출력 파일 경로 설정
    if output_file is None:
        csv_path = Path(csv_file)
        output_file = csv_path.parent / f"{csv_path.stem}_contour_2d.png"
    
    # 그래프 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"등고선 그래프 저장 완료: {output_file}")
    
    # 범위 정보 출력
    print(f"\n값 범위 정보:")
    print(f"  X: {x_min:.6f} ~ {x_max:.6f} (범위: {x_range:.6f})")
    print(f"  Y: {y_min:.6f} ~ {y_max:.6f} (범위: {y_range:.6f})")
    print(f"  Z: {z_min:.6f} ~ {z_max:.6f} (범위: {z_range:.6f})")
    print(f"  각 그래프는 개별 범위로 표시됩니다.")
    
    # 그래프 창 표시하지 않고 닫기
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='머지된 전주 데이터를 등고선 2D 그래프로 시각화')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_FILE,
                       help=f'머지된 CSV 파일 경로 (기본값: {DEFAULT_CSV_FILE})')
    parser.add_argument('--output', type=str, default=None, help='출력 이미지 파일 경로 (기본값: CSV 파일명_contour_2d.png)')
    parser.add_argument('--break-info', type=str, default=None, help='break_info.json 파일 경로 (파단 위치 표시용)')
    
    args = parser.parse_args()
    
    # 명령줄 인자로 파일이 지정되지 않았으면 DEFAULT_CSV_FILE 사용
    csv_file = args.csv if args.csv != DEFAULT_CSV_FILE or len(sys.argv) > 1 else DEFAULT_CSV_FILE
    
    # break_info.json 파일 읽기
    break_info = None
    if args.break_info and os.path.exists(args.break_info):
        import json
        try:
            with open(args.break_info, 'r', encoding='utf-8') as f:
                break_info = json.load(f)
        except Exception as e:
            print(f"경고: break_info.json 읽기 실패: {e}")
    
    plot_contour_2d(csv_file, args.output, break_info)

if __name__ == "__main__":
    main()

