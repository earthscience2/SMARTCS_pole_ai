#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 파일에서 2D 등고선 플롯을 생성하는 모듈
x, y, z 값에 대한 3개의 서브플롯 생성 (컬러바 없음)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안 함
import matplotlib.pyplot as plt

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    try:
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    except:
        try:
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        except:
            pass
elif platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'AppleGothic'
else:
    matplotlib.rcParams['font.family'] = 'NanumGothic'

matplotlib.rcParams['axes.unicode_minus'] = False


def plot_csv_2d(csv_file, output_file=None, break_info_file=None):
    """
    CSV 파일에서 2D 등고선 플롯 생성 (x, y, z 3개 서브플롯, 컬러바 없음)
    
    Args:
        csv_file: CSV 파일 경로 (height, degree, x_value, y_value, z_value 포함)
        output_file: 출력 이미지 파일 경로 (None이면 CSV 파일과 같은 위치에 저장)
        break_info_file: 파단 정보 파일 경로 (사용하지 않음, 호환성을 위해 유지)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        
        if df.empty:
            return
        
        # 필요한 컬럼 확인
        required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return
        
        # 데이터를 그리드로 변환
        heights = sorted(df['height'].unique())
        degrees = sorted(df['degree'].unique())
        
        # pivot table 생성
        x_grid = df.pivot_table(index='height', columns='degree', values='x_value', aggfunc='mean')
        y_grid = df.pivot_table(index='height', columns='degree', values='y_value', aggfunc='mean')
        z_grid = df.pivot_table(index='height', columns='degree', values='z_value', aggfunc='mean')
        
        # 인덱스와 컬럼 정렬
        x_grid = x_grid.reindex(index=heights, columns=degrees)
        y_grid = y_grid.reindex(index=heights, columns=degrees)
        z_grid = z_grid.reindex(index=heights, columns=degrees)
        
        # YOLO 학습을 위해 원본 데이터 특징을 유지하도록 원본 그리드 사용
        # 메쉬그리드 생성 (원본 그리드 사용)
        D, H = np.meshgrid(degrees, heights)
        
        # 그림 생성 (3개 서브플롯)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 각 서브플롯에 등고선 플롯
        grids = [x_grid, y_grid, z_grid]
        titles = ['X Value', 'Y Value', 'Z Value']
        
        for idx, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
            # 원본 그리드 값 사용 (YOLO 학습을 위해 데이터 특징 유지)
            grid_values = grid.values
            
            # 유효한 값으로 범위 계산
            valid_values = grid_values[~np.isnan(grid_values)]
            
            if len(valid_values) > 0:
                # 데이터의 수치 변화를 더 잘 보이도록 범위 설정 개선
                # 중앙값과 표준편차 기반으로 범위 설정
                median_val = np.median(valid_values)
                std_val = np.std(valid_values)
                mean_val = np.mean(valid_values)
                
                # 변화를 더 잘 보이도록 범위를 좁게 설정 (표준편차의 2배 사용)
                # 이렇게 하면 작은 변화도 더 뚜렷하게 보임
                vmin = median_val - 2.5 * std_val
                vmax = median_val + 2.5 * std_val
                
                # 실제 데이터 범위 확인
                data_min = np.min(valid_values)
                data_max = np.max(valid_values)
                data_range = data_max - data_min
                
                # percentile 기반 범위 (더 보수적인 범위)
                p5 = np.percentile(valid_values, 2)  # 2% percentile
                p95 = np.percentile(valid_values, 98)  # 98% percentile
                
                # 데이터 범위가 넓으면 percentile 사용, 좁으면 표준편차 기반 사용
                if data_range > 0:
                    # 변화를 강조하기 위해 범위를 조정
                    # 작은 변화도 잘 보이도록 범위를 적절히 설정
                    if std_val > 0 and data_range / std_val > 3:
                        # 데이터가 넓게 분포되어 있으면 percentile 사용
                        vmin = p5
                        vmax = p95
                    else:
                        # 데이터가 좁게 분포되어 있으면 표준편차 기반 사용
                        vmin = max(data_min, median_val - 2.5 * std_val)
                        vmax = min(data_max, median_val + 2.5 * std_val)
                else:
                    vmin = data_min
                    vmax = data_max
                
                # 범위가 너무 좁으면 전체 범위 사용
                if vmax - vmin < 1e-6:
                    vmin = data_min
                    vmax = data_max
            else:
                vmin = None
                vmax = None
            
            # 등고선 레벨 수를 늘려서 등고선을 더 촘촘하게 표현
            # 레벨이 많을수록 더 세밀한 변화를 표현할 수 있음
            num_levels = 30
            
            # 빨강-파랑 계열 컬러맵 사용
            # 'RdBu': 빨강(Red)에서 파랑(Blue)으로 변화 (낮은 값=빨강, 높은 값=파랑)
            # 'coolwarm': 파랑에서 빨강으로 (낮은 값=파랑, 높은 값=빨강)
            # 'seismic': 파랑-흰색-빨강 (중간값=흰색)
            # 빨강과 파랑을 모두 포함한 'RdBu' 사용
            contour = ax.contourf(D, H, grid_values, levels=num_levels, cmap='RdBu_r', 
                                 vmin=vmin, vmax=vmax, antialiased=True, extend='both')
            
            # 등고선 라인 추가하여 변화 구간을 더 명확하게 표시
            contour_lines = ax.contour(D, H, grid_values, levels=num_levels, 
                                       colors='black', linewidths=0.5, alpha=0.3)
            
            ax.set_xlabel('Degree')
            ax.set_ylabel('Height (m)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # 출력 파일 경로 결정
        if output_file is None:
            output_file = csv_file.replace('.csv', '_2d_plot.png')
        
        # 이미지 저장
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"이미지 생성 오류: {e}")
        import traceback
        traceback.print_exc()
