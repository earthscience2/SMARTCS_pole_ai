#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
합쳐진 전주 데이터 파일을 3D로 시각화하는 스크립트

사용법:
    python plot_merged_pole_data_3d.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

def plot_3d_surface(csv_file_path, output_dir=None):
    """
    CSV 파일의 데이터를 3D 표면 플롯으로 시각화
    새로운 구조: height, degree, x_value, y_value, z_value
    
    Args:
        csv_file_path: CSV 파일 경로
        output_dir: 출력 디렉토리 (None이면 CSV 파일과 같은 디렉토리에 저장)
    """
    if not os.path.exists(csv_file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    
    # CSV 파일 읽기
    print(f"파일 읽는 중: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("오류: 데이터가 비어있습니다.")
        return
    
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    
    # height와 degree 컬럼 확인
    if 'height' not in df.columns or 'degree' not in df.columns:
        print("오류: 'height' 또는 'degree' 컬럼이 없습니다.")
        return
    
    # x_value, y_value, z_value 컬럼 확인
    value_cols = {}
    for axis in ['x', 'y', 'z']:
        col_name = f'{axis}_value'
        if col_name in df.columns:
            value_cols[axis] = col_name
    
    if not value_cols:
        print("오류: x_value, y_value, z_value 컬럼을 찾을 수 없습니다.")
        return
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    poleid = os.path.basename(csv_file_path).replace('_OUT_merged.csv', '').replace('_merged.csv', '')
    
    # 높이와 각도로 그리드 생성
    heights = sorted(df['height'].unique())
    degrees = sorted(df['degree'].unique())
    
    if len(heights) == 0 or len(degrees) == 0:
        print("경고: 높이 또는 각도 데이터가 없습니다.")
        return
    
    print(f"\n높이 범위: {min(heights):.1f}~{max(heights):.1f}m ({len(heights)}개)")
    print(f"각도 범위: {min(degrees):.0f}~{max(degrees):.0f}도 ({len(degrees)}개)")
    
    # 각 축(x, y, z)별로 3D 서페이스 플롯 생성
    for axis in ['x', 'y', 'z']:
        if axis not in value_cols:
            continue
        
        col_name = value_cols[axis]
        print(f"\n{axis.upper()} 축 데이터 처리 중...")
        
        # 높이와 각도 그리드에 값 매핑
        values = np.full((len(heights), len(degrees)), np.nan)
        
        for i, h in enumerate(heights):
            for j, d in enumerate(degrees):
                mask = (df['height'] == h) & (df['degree'] == d)
                if mask.any():
                    val = df.loc[mask, col_name].iloc[0]
                    if pd.notna(val):
                        values[i, j] = val
        
        # NaN이 너무 많으면 스킵
        if np.isnan(values).all():
            print(f"  경고: {axis.upper()} 축 데이터가 모두 NaN입니다.")
            continue
        
        # 3D 플롯 생성
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 메쉬 그리드 생성
        H, D = np.meshgrid(degrees, heights)
        
        # 서페이스 플롯
        surf = ax.plot_surface(H, D, values, cmap=cm.viridis, 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # 컬러바 추가
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label=f'{axis.upper()} Value')
        
        # 축 레이블 설정
        ax.set_xlabel('Degree (°)', fontsize=12, labelpad=10)
        ax.set_ylabel('Height (m)', fontsize=12, labelpad=10)
        ax.set_zlabel(f'{axis.upper()} Value', fontsize=12, labelpad=10)
        
        # 제목 설정
        ax.set_title(f'{poleid} - {axis.upper()} Axis\n3D Surface Plot', 
                    fontsize=14, pad=20)
        
        # 저장
        output_filename = f"{poleid}_{axis}_value_3d.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  저장: {output_path}")
        plt.close()
    
    # x, y, z 값을 하나의 3D 공간에 3개의 평면으로 표시
    print(f"\n통합 3D 플롯 생성 중 (x, y, z 평면)...")
    
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # 각 축별로 그리드 생성
    x_values = np.full((len(heights), len(degrees)), np.nan)
    y_values = np.full((len(heights), len(degrees)), np.nan)
    z_values = np.full((len(heights), len(degrees)), np.nan)
    
    for i, h in enumerate(heights):
        for j, d in enumerate(degrees):
            mask = (df['height'] == h) & (df['degree'] == d)
            if mask.any():
                row = df.loc[mask].iloc[0]
                if 'x_value' in value_cols and pd.notna(row.get('x_value')):
                    x_values[i, j] = row['x_value']
                if 'y_value' in value_cols and pd.notna(row.get('y_value')):
                    y_values[i, j] = row['y_value']
                if 'z_value' in value_cols and pd.notna(row.get('z_value')):
                    z_values[i, j] = row['z_value']
    
    H, D = np.meshgrid(degrees, heights)
    
    # x, y, z 값을 각각 다른 평면으로 표시 (반투명하게)
    # X 값 - 빨간색 계열
    if not np.isnan(x_values).all():
        surf_x = ax.plot_surface(H, D, x_values, cmap=cm.Reds, 
                                 alpha=0.5, linewidth=0, antialiased=True,
                                 vmin=np.nanmin(x_values), vmax=np.nanmax(x_values))
        # 컬러바 추가
        fig.colorbar(surf_x, ax=ax, shrink=0.4, aspect=15, pad=0.05, label='X Value')
    
    # Y 값 - 초록색 계열
    if not np.isnan(y_values).all():
        surf_y = ax.plot_surface(H, D, y_values, cmap=cm.Greens, 
                                 alpha=0.5, linewidth=0, antialiased=True,
                                 vmin=np.nanmin(y_values), vmax=np.nanmax(y_values))
        # 컬러바 추가
        fig.colorbar(surf_y, ax=ax, shrink=0.4, aspect=15, pad=0.15, label='Y Value')
    
    # Z 값 - 파란색 계열
    if not np.isnan(z_values).all():
        surf_z = ax.plot_surface(H, D, z_values, cmap=cm.Blues, 
                                 alpha=0.5, linewidth=0, antialiased=True,
                                 vmin=np.nanmin(z_values), vmax=np.nanmax(z_values))
        # 컬러바 추가
        fig.colorbar(surf_z, ax=ax, shrink=0.4, aspect=15, pad=0.25, label='Z Value')
    
    # 축 레이블 설정
    ax.set_xlabel('Degree (°)', fontsize=14, labelpad=15)
    ax.set_ylabel('Height (m)', fontsize=14, labelpad=15)
    ax.set_zlabel('Value', fontsize=14, labelpad=15)
    
    # 제목 설정
    ax.set_title(f'{poleid} - X, Y, Z Values in 3D Space\n(Red: X-axis, Green: Y-axis, Blue: Z-axis)', 
                fontsize=16, pad=25)
    
    # 그리드 표시
    ax.grid(True, alpha=0.3)
    
    # 저장
    output_filename = f"{poleid}_xyz_combined_3d.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  저장: {output_path}")
    plt.close()
    
    print("\n3D 시각화 완료!")

def plot_3d_scatter(csv_file_path, output_dir=None, channel='ch1', axis_type='x'):
    """
    CSV 파일의 데이터를 3D 스캐터 플롯으로 시각화
    
    Args:
        csv_file_path: CSV 파일 경로
        output_dir: 출력 디렉토리
        channel: 시각화할 채널 (예: 'ch1', 'x_ch1')
        axis_type: 축 타입 ('x', 'y', 'z')
    """
    if not os.path.exists(csv_file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {csv_file_path}")
        return
    
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("오류: 데이터가 비어있습니다.")
        return
    
    if channel not in df.columns:
        print(f"오류: '{channel}' 컬럼이 없습니다.")
        return
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 필터링 (NaN 제외)
    df_clean = df[['height', 'degree', channel]].dropna()
    
    if df_clean.empty:
        print(f"오류: '{channel}' 유효한 데이터가 없습니다.")
        return
    
    # 3D 스캐터 플롯
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df_clean['degree'], df_clean['height'], df_clean[channel],
                        c=df_clean[channel], cmap=cm.viridis, s=10, alpha=0.6)
    
    # 컬러바
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Sensor Value')
    
    # 축 레이블
    ax.set_xlabel('Degree (°)', fontsize=12, labelpad=10)
    ax.set_ylabel('Height (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Sensor Value', fontsize=12, labelpad=10)
    
    # 제목
    poleid = os.path.basename(csv_file_path).replace('_OUT_merged.csv', '').replace('_merged.csv', '')
    ax.set_title(f'{poleid} - {axis_type.upper()} Axis - {channel.upper()}\n3D Scatter Plot', 
                fontsize=14, pad=20)
    
    # 저장
    output_filename = f"{poleid}_{axis_type}_{channel}_3d_scatter.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"저장: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='합쳐진 전주 데이터를 3D로 시각화')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_FILE,
                       help=f'CSV 파일 경로 (기본값: {DEFAULT_CSV_FILE})')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 디렉토리 (기본값: CSV 파일과 같은 디렉토리)')
    parser.add_argument('--mode', type=str, default='surface', choices=['surface', 'scatter'],
                       help='시각화 모드: surface (서페이스) 또는 scatter (스캐터)')
    
    args = parser.parse_args()
    
    # 명령줄 인자로 파일이 지정되지 않았으면 DEFAULT_CSV_FILE 사용
    csv_file = args.csv if args.csv != DEFAULT_CSV_FILE or len(sys.argv) > 1 else DEFAULT_CSV_FILE
    
    print("=" * 60)
    print("전주 데이터 3D 시각화 시작")
    print("=" * 60)
    print(f"파일: {csv_file}")
    print(f"모드: {args.mode}")
    print("=" * 60)
    
    if args.mode == 'surface':
        plot_3d_surface(csv_file, args.output)
    else:
        # 스캐터 모드는 x_value만 시각화 (예제)
        plot_3d_scatter(csv_file, args.output, channel='x_value', axis_type='x')
    
    print("\n" + "=" * 60)
    print("3D 시각화 완료")
    print("=" * 60)

