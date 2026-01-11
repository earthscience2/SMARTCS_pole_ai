#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
크롭된 파단 영역 CSV 파일과 정상 데이터를 시퀀스 데이터로 변환하여 학습 데이터를 준비하는 스크립트.

두 가지 모드 지원:
1. 기존 방식 (--sliding-window 미사용): 크롭된 CSV 파일에서 직접 시퀀스 생성
2. 슬라이딩 윈도우 방식 (--sliding-window 사용): 평가 시와 동일한 방식으로 슬라이딩 윈도우 생성

입력: 
  - 기존 방식: 5. crop_data/break/, 5. crop_data/normal/ 디렉토리
  - 슬라이딩 윈도우 방식: 4. edit_pole_data/break/, 4. edit_pole_data/normal/ 디렉토리
출력: 6. train_data/sequences/ 디렉토리에 시퀀스 데이터 및 메타데이터 저장

시퀀스 형식:
- 각 샘플: (sequence_length, 3) 형태
- 피처: [x_value, y_value, z_value] (각도와 높이는 정렬에만 사용, 학습 피처에서는 제외)
- 높이 기준 정렬 후 시퀀스 생성
- 파단 데이터: 라벨 1
- 정상 데이터: 라벨 0 (파일당 여러 샘플 생성 가능)

슬라이딩 윈도우 방식:
- 전체 CSV 파일에서 슬라이딩 윈도우 생성 (평가 시와 동일)
- 파단 데이터: 윈도우가 파단 위치를 포함하면 라벨 1, 아니면 라벨 0
- 정상 데이터: 모든 윈도우는 라벨 0
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import re

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_crop_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    크롭된 CSV 파일 로드.
    
    Args:
        csv_path: CSV 파일 경로
    
    Returns:
        DataFrame 또는 None (로드 실패 시)
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  경고: 필수 컬럼이 없습니다 {missing_cols}: {csv_path}")
            return None
        
        return df
    except Exception as e:
        print(f"  경고: CSV 로드 실패 ({csv_path}): {e}")
        return None


def prepare_sequence_from_csv(
    csv_path: str,
    sequence_length: int = 50,
    sort_by: str = 'height',
    normalize: bool = True
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    CSV 파일에서 시퀀스 데이터를 생성.
    
    Args:
        csv_path: CSV 파일 경로
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        normalize: 정규화 여부
    
    Returns:
        tuple: (시퀀스 배열, 메타데이터) 또는 None
    """
    df = load_crop_csv(csv_path)
    if df is None:
        return None
    
    # 높이 또는 각도 기준 정렬 (정렬에만 사용, 학습 피처에는 포함하지 않음)
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    
    # 피처 선택: [x, y, z] (각도와 높이는 파단과의 연관성이 없어 학습 피처에서 제외)
    features = df[['x_value', 'y_value', 'z_value']].values
    
    # 정규화
    scaler_info = None
    if normalize:
        scaler = RobustScaler()  # 이상치에 강건한 스케일러
        features = scaler.fit_transform(features)
        scaler_info = {
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist(),
        }
    
    # 시퀀스 길이 맞추기
    n_points = len(features)
    
    if n_points < sequence_length:
        # 패딩 (마지막 값으로 채우기)
        padding = np.tile(features[-1:], (sequence_length - n_points, 1))
        sequence = np.vstack([features, padding])
    elif n_points > sequence_length:
        # 자르기 (중간 부분 선택 또는 균등하게 샘플링)
        indices = np.linspace(0, n_points - 1, sequence_length, dtype=int)
        sequence = features[indices]
    else:
        sequence = features
    
    # 메타데이터
    metadata = {
        'original_length': int(n_points),
        'sequence_length': int(sequence_length),
        'height_min': float(df['height'].min()),
        'height_max': float(df['height'].max()),
        'degree_min': float(df['degree'].min()),
        'degree_max': float(df['degree'].max()),
        'scaler_info': scaler_info,
    }
    
    return sequence, metadata


def collect_all_crop_files(crop_base_dir: str) -> List[Tuple[str, str, str, int]]:
    """
    모든 크롭된 CSV 파일 수집 (파단 데이터).
    
    Args:
        crop_base_dir: 크롭된 파일이 있는 기본 디렉토리
    
    Returns:
        list: [(csv_path, project_name, poleid, label), ...] 리스트 (label=1)
    """
    crop_path = Path(current_dir) / crop_base_dir
    crop_files = []
    
    if not crop_path.exists():
        print(f"경고: 크롭 디렉토리를 찾을 수 없습니다: {crop_base_dir}")
        return crop_files
    
    # 모든 프로젝트 디렉토리 찾기
    for project_dir in crop_path.iterdir():
        if not project_dir.is_dir():
            continue
        
        project_name = project_dir.name
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            
            poleid = pole_dir.name
            
            # 크롭된 CSV 파일 찾기
            for csv_file in pole_dir.glob("*_break_crop.csv"):
                crop_files.append((str(csv_file), project_name, poleid, 1))  # 라벨 1: 파단
    
    return crop_files


def collect_normal_data_files(normal_data_dir: str, use_cropped: bool = True) -> List[Tuple[str, str, str, int]]:
    """
    정상 데이터 파일 수집 (크롭된 파일 우선 사용).
    
    Args:
        normal_data_dir: 정상 데이터 디렉토리
        use_cropped: 크롭된 파일 사용 여부 (True면 *_normal_crop.csv, False면 *_processed.csv)
    
    Returns:
        list: [(csv_path, project_name, poleid, label), ...] 리스트 (label=0)
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(normal_data_dir):
        normal_path = Path(current_dir) / normal_data_dir
    else:
        normal_path = Path(normal_data_dir)
    
    normal_files = []
    
    if not normal_path.exists():
        print(f"경고: 정상 데이터 디렉토리를 찾을 수 없습니다: {normal_data_dir}")
        print(f"  시도한 경로: {normal_path}")
        print(f"  현재 디렉토리: {current_dir}")
        return normal_files
    
    # 모든 프로젝트 디렉토리 찾기
    for project_dir in normal_path.iterdir():
        if not project_dir.is_dir():
            continue
        
        project_name = project_dir.name
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            
            poleid = pole_dir.name
            
            # 크롭된 파일 우선 사용
            if use_cropped:
                csv_files = list(pole_dir.glob("*_normal_crop.csv"))
                # 크롭된 파일이 없으면 processed 파일 사용
                if not csv_files:
                    csv_files = list(pole_dir.glob("*_processed.csv"))
            else:
                csv_files = list(pole_dir.glob("*_processed.csv"))
            
            # CSV 파일 추가 (크롭된 파일은 이미 크롭되어 있으므로 샘플링 불필요)
            for csv_file in csv_files:
                normal_files.append((str(csv_file), project_name, poleid, 0))  # 라벨 0: 정상
    
    return normal_files


def prepare_sequence_from_normal_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    sort_by: str = 'height',
    normalize: bool = True,
    sample_index: int = 0,
    max_samples: int = 3
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    정상 데이터 DataFrame에서 시퀀스 데이터 생성 (높이 구간별 샘플링).
    
    Args:
        df: DataFrame
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        normalize: 정규화 여부
        sample_index: 샘플 인덱스 (0부터 max_samples-1)
        max_samples: 파일당 최대 샘플 수
    
    Returns:
        tuple: (시퀀스 배열, 메타데이터) 또는 None
    """
    if df.empty:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    # 정렬
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    
    n_points = len(df)
    
    # 높이 구간별로 샘플링
    if max_samples > 1 and n_points >= sequence_length * max_samples:
        # 높이를 구간으로 나누기
        height_min, height_max = df['height'].min(), df['height'].max()
        height_range = height_max - height_min
        
        # 샘플 구간 계산
        sample_height_min = height_min + (height_range / max_samples) * sample_index
        sample_height_max = height_min + (height_range / max_samples) * (sample_index + 1)
        
        # 해당 구간의 데이터 필터링
        sample_df = df[(df['height'] >= sample_height_min) & (df['height'] <= sample_height_max)].copy()
        
        if len(sample_df) < 5:  # 최소 포인트 수 미만이면 전체 데이터 사용
            sample_df = df.copy()
    else:
        sample_df = df.copy()
    
    # 피처 선택: [x, y, z]
    features = sample_df[['x_value', 'y_value', 'z_value']].values
    
    # 정규화
    scaler_info = None
    if normalize:
        scaler = RobustScaler()
        features = scaler.fit_transform(features)
        scaler_info = {
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist(),
        }
    
    # 시퀀스 길이 맞추기
    n_points = len(features)
    
    if n_points < sequence_length:
        # 패딩
        padding = np.tile(features[-1:], (sequence_length - n_points, 1))
        sequence = np.vstack([features, padding])
    elif n_points > sequence_length:
        # 균등 샘플링
        indices = np.linspace(0, n_points - 1, sequence_length, dtype=int)
        sequence = features[indices]
    else:
        sequence = features
    
    # 메타데이터
    metadata = {
        'original_length': int(n_points),
        'sequence_length': int(sequence_length),
        'height_min': float(sample_df['height'].min()),
        'height_max': float(sample_df['height'].max()),
        'degree_min': float(sample_df['degree'].min()),
        'degree_max': float(sample_df['degree'].max()),
        'scaler_info': scaler_info,
    }
    
    return sequence, metadata


def prepare_sequence_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    sort_by: str = 'height',
    scaler: Optional[RobustScaler] = None
) -> Optional[np.ndarray]:
    """
    DataFrame에서 시퀀스 데이터 생성 (8. detect_break_with_lstm.py와 동일한 방식).
    
    Args:
        df: DataFrame (height, degree, x_value, y_value, z_value 포함)
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        scaler: 스케일러 (None이면 새로 생성하여 fit_transform)
    
    Returns:
        시퀀스 배열 또는 None
    """
    if df.empty:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    # 정렬
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    
    # 피처 선택: [x, y, z]
    features = df[['x_value', 'y_value', 'z_value']].values
    
    # 정규화
    if scaler is not None:
        features = scaler.transform(features)
    else:
        # 새로 생성하여 fit_transform
        scaler = RobustScaler()
        features = scaler.fit_transform(features)
    
    # 시퀀스 길이 맞추기
    n_points = len(features)
    
    if n_points < sequence_length:
        # 패딩
        padding = np.tile(features[-1:], (sequence_length - n_points, 1))
        sequence = np.vstack([features, padding])
    elif n_points > sequence_length:
        # 균등 샘플링
        indices = np.linspace(0, n_points - 1, sequence_length, dtype=int)
        sequence = features[indices]
    else:
        sequence = features
    
    return sequence.reshape(1, sequence_length, -1)[0]  # (seq_len, n_features)


def check_window_contains_break(
    height_start: float,
    height_end: float,
    degree_start: float,
    degree_end: float,
    break_height: Optional[float],
    break_degree: Optional[float],
    height_margin: float = 0.15
) -> bool:
    """
    윈도우가 파단 위치를 포함하는지 확인.
    
    Args:
        height_start, height_end: 윈도우 높이 범위
        degree_start, degree_end: 윈도우 각도 범위
        break_height: 파단 높이 (None이면 확인 안 함)
        break_degree: 파단 각도 (None이면 확인 안 함)
        height_margin: 파단 높이 기준 ± margin
    
    Returns:
        bool: 파단 위치를 포함하면 True
    """
    if break_height is None:
        return False
    
    # 높이 범위 확인 (break_height ± margin이 윈도우와 겹치는지)
    break_h_min = break_height - height_margin
    break_h_max = break_height + height_margin
    
    height_overlap = not (height_end < break_h_min or height_start > break_h_max)
    
    if not height_overlap:
        return False
    
    # 각도 범위 확인 (break_degree가 윈도우 각도 범위 안에 있는지)
    if break_degree is not None:
        def is_degree_in_range(deg, d_min, d_max):
            if d_min > d_max:  # 0도 경계를 넘어가는 경우 (예: 350~10도)
                return deg >= d_min or deg <= d_max
            else:
                return d_min <= deg <= d_max
        
        degree_contained = is_degree_in_range(break_degree, degree_start, degree_end)
        return degree_contained
    
    return True


def create_sliding_windows_from_csv(
    csv_path: str,
    break_info: Optional[Dict] = None,
    window_height: float = 0.3,
    window_degree: float = 90.0,
    stride_height: float = 0.1,
    stride_degree: float = 30.0,
    sequence_length: int = 50,
    sort_by: str = 'height'
) -> List[Tuple[np.ndarray, Dict, int]]:
    """
    CSV 파일에서 슬라이딩 윈도우 방식으로 시퀀스 생성 (평가 시와 동일한 방식).
    
    Args:
        csv_path: CSV 파일 경로
        break_info: 파단 정보 딕셔너리 (break_info.json 내용, None이면 정상 데이터로 간주)
        window_height: 윈도우 높이 범위
        window_degree: 윈도우 각도 범위
        stride_height: 높이 스트라이드
        stride_degree: 각도 스트라이드
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준
    
    Returns:
        [(시퀀스, 메타데이터, 라벨), ...] 리스트
    """
    # CSV 로드
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'height' not in df.columns or 'degree' not in df.columns:
            return []
    except Exception:
        return []
    
    required_cols = ['x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return []
    
    # 파단 정보 추출
    break_height = None
    break_degree = None
    if break_info and break_info.get('breakstate') == 'B':
        break_height = break_info.get('breakheight')
        break_degree = break_info.get('breakdegree')
        
        if break_height is not None:
            try:
                break_height = float(break_height)
                if break_height >= 2.0:
                    break_height = None  # 2m 이상이면 무시
            except (ValueError, TypeError):
                break_height = None
        
        if break_degree is not None:
            try:
                break_degree = float(break_degree)
            except (ValueError, TypeError):
                break_degree = None
    
    sequences = []
    h_min, h_max = df['height'].min(), df['height'].max()
    d_min, d_max = df['degree'].min(), df['degree'].max()
    
    # 윈도우 생성
    h_starts = np.arange(h_min, h_max - window_height + stride_height, stride_height)
    
    # 각도 슬라이딩: CSV의 각도 범위가 윈도우 각도와 같거나 작으면 슬라이딩 불필요
    # 각 CSV 파일은 보통 90도 범위를 가지므로 각도 슬라이딩은 의미 없음
    degree_range = d_max - d_min
    if degree_range <= window_degree:
        # 각도 범위가 윈도우 크기 이하면 윈도우 1개만 생성
        d_starts = [d_min]
    else:
        # 각도 범위가 윈도우 크기보다 크면 슬라이딩 적용
        d_starts = np.arange(d_min, d_max - window_degree + stride_degree, stride_degree)
    
    scaler = RobustScaler()
    # 전체 데이터로 scaler fit (평가 시와 동일하게)
    features_all = df[['x_value', 'y_value', 'z_value']].values
    scaler.fit(features_all)
    
    for h_start in h_starts:
        h_end = h_start + window_height
        for d_start in d_starts:
            d_end = d_start + window_degree
            
            # 윈도우 내 데이터 필터링
            window_df = df[
                (df['height'] >= h_start) & (df['height'] <= h_end) &
                (df['degree'] >= d_start) & (df['degree'] <= d_end)
            ].copy()
            
            if len(window_df) < 5:  # 최소 포인트 수
                continue
            
            # 시퀀스 생성
            sequence = prepare_sequence_from_dataframe(
                window_df,
                sequence_length=sequence_length,
                sort_by=sort_by,
                scaler=scaler  # 이미 fit된 scaler 사용
            )
            
            if sequence is None:
                continue
            
            # 라벨 결정
            if break_info and break_height is not None:
                # 파단 데이터: 윈도우가 파단 위치를 포함하는지 확인
                label = 1 if check_window_contains_break(
                    h_start, h_end, d_start, d_end,
                    break_height, break_degree
                ) else 0
            else:
                # 정상 데이터: 모든 윈도우는 라벨 0
                label = 0
            
            metadata = {
                'height_start': float(h_start),
                'height_end': float(h_end),
                'degree_start': float(d_start),
                'degree_end': float(d_end),
                'center_height': float((h_start + h_end) / 2),
                'center_degree': float((d_start + d_end) / 2),
                'num_points': int(len(window_df))
            }
            
            sequences.append((sequence, metadata, label))
    
    return sequences


def load_break_info_json(pole_dir: Path) -> Optional[Dict]:
    """
    break_info.json 로드.
    
    Args:
        pole_dir: 전주 디렉토리 경로
    
    Returns:
        break_info 딕셔너리 또는 None
    """
    break_info_path = pole_dir / f"{pole_dir.name}_break_info.json"
    if not break_info_path.exists():
        return None
    
    try:
        with open(break_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def process_with_sliding_window(
    edit_data_dir: str = "4. edit_pole_data",
    normal_data_dir: str = "4. edit_pole_data/normal",
    output_dir: str = "6. train_data/sequences",
    sequence_length: int = 50,
    sort_by: str = 'height',
    window_height: float = 0.3,
    window_degree: float = 90.0,
    stride_height: float = 0.1,
    stride_degree: float = 30.0
):
    """
    슬라이딩 윈도우 방식으로 시퀀스 데이터 준비 (평가 시와 동일한 방식).
    
    Args:
        edit_data_dir: edit_pole_data 디렉토리 경로
        normal_data_dir: 정상 데이터 디렉토리
        output_dir: 출력 디렉토리
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준
        window_height: 윈도우 높이 범위
        window_degree: 윈도우 각도 범위
        stride_height: 높이 스트라이드
        stride_degree: 각도 스트라이드
    """
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 파단 데이터 파일 수집 (4. edit_pole_data/break/)
    break_base_dir = Path(current_dir) / edit_data_dir / "break"
    break_files = []
    
    if break_base_dir.exists():
        print("파단 데이터 파일 수집 중 (슬라이딩 윈도우 모드)...")
        for project_dir in break_base_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_name = project_dir.name
            
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                poleid = pole_dir.name
                
                # *_OUT_processed.csv 파일 찾기
                for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                    break_files.append((str(csv_file), project_name, poleid, pole_dir))
    
    print(f"  파단 파일: {len(break_files)}개")
    
    # 정상 데이터 파일 수집 (4. edit_pole_data/normal/)
    normal_base_dir = Path(current_dir) / normal_data_dir
    normal_files = []
    
    if normal_base_dir.exists():
        print(f"\n정상 데이터 파일 수집 중...")
        for project_dir in normal_base_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_name = project_dir.name
            
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                poleid = pole_dir.name
                
                # *_OUT_processed.csv 또는 *_normal_processed.csv 파일 찾기
                for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                    normal_files.append((str(csv_file), project_name, poleid))
                for csv_file in pole_dir.glob("*_normal_processed.csv"):
                    normal_files.append((str(csv_file), project_name, poleid))
    
    print(f"  정상 파일: {len(normal_files)}개")
    
    if not break_files and not normal_files:
        print("처리할 파일을 찾을 수 없습니다.")
        return
    
    # 시퀀스 데이터 처리
    sequences = []
    labels = []
    metadata_list = []
    failed_files = []
    
    # 파단 데이터 처리
    if break_files:
        print(f"\n파단 데이터 슬라이딩 윈도우 생성 중...")
        print(f"  윈도우 크기: 높이 {window_height}m, 각도 {window_degree}°")
        print(f"  스트라이드: 높이 {stride_height}m, 각도 {stride_degree}°")
        
        for csv_path, project_name, poleid, pole_dir in tqdm(break_files, desc="파단 데이터 처리"):
            break_info = load_break_info_json(pole_dir)
            
            window_sequences = create_sliding_windows_from_csv(
                csv_path=csv_path,
                break_info=break_info,
                window_height=window_height,
                window_degree=window_degree,
                stride_height=stride_height,
                stride_degree=stride_degree,
                sequence_length=sequence_length,
                sort_by=sort_by
            )
            
            if not window_sequences:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '슬라이딩 윈도우 생성 실패'
                })
                continue
            
            for sequence, metadata, label in window_sequences:
                sequences.append(sequence)
                labels.append(label)
                metadata['csv_path'] = csv_path
                metadata['project_name'] = project_name
                metadata['poleid'] = poleid
                metadata_list.append(metadata)
    
    # 정상 데이터 처리
    if normal_files:
        print(f"\n정상 데이터 슬라이딩 윈도우 생성 중...")
        for csv_path, project_name, poleid in tqdm(normal_files, desc="정상 데이터 처리"):
            window_sequences = create_sliding_windows_from_csv(
                csv_path=csv_path,
                break_info=None,  # 정상 데이터는 break_info 없음
                window_height=window_height,
                window_degree=window_degree,
                stride_height=stride_height,
                stride_degree=stride_degree,
                sequence_length=sequence_length,
                sort_by=sort_by
            )
            
            if not window_sequences:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '슬라이딩 윈도우 생성 실패'
                })
                continue
            
            for sequence, metadata, label in window_sequences:
                sequences.append(sequence)
                labels.append(label)  # 항상 0
                metadata['csv_path'] = csv_path
                metadata['project_name'] = project_name
                metadata['poleid'] = poleid
                metadata_list.append(metadata)
    
    if not sequences:
        print("생성된 시퀀스 데이터가 없습니다.")
        return
    
    # 배열로 변환
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"\n시퀀스 데이터 생성 완료:")
    print(f"  총 샘플 수: {len(X)}")
    print(f"  시퀀스 형태: {X.shape}")
    print(f"  실패한 파일: {len(failed_files)}개")
    
    # 데이터 저장
    sequences_file = output_path / "break_sequences.npy"
    labels_file = output_path / "break_labels.npy"
    metadata_file = output_path / "break_sequences_metadata.json"
    
    np.save(sequences_file, X)
    np.save(labels_file, y)
    
    # 메타데이터 저장
    metadata_dict = {
        'total_samples': len(metadata_list),
        'sequence_length': sequence_length,
        'feature_names': ['x_value', 'y_value', 'z_value'],
        'sort_by': sort_by,
        'use_sliding_window': True,
        'window_height': window_height,
        'window_degree': window_degree,
        'stride_height': stride_height,
        'stride_degree': stride_degree,
        'samples': metadata_list,
        'failed_files': failed_files,
        'data_shape': list(X.shape),
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터 저장 완료:")
    print(f"  시퀀스 데이터: {sequences_file}")
    print(f"  라벨 데이터: {labels_file}")
    print(f"  메타데이터: {metadata_file}")
    
    # 통계 정보 출력
    break_samples = [i for i, label in enumerate(labels) if label == 1]
    normal_samples = [i for i, label in enumerate(labels) if label == 0]
    
    print(f"\n데이터 통계:")
    print(f"  총 샘플 수: {len(sequences)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  파단 샘플 비율: {len(break_samples)/len(sequences)*100:.2f}%")


def process_all_crops(
    crop_base_dir: str = "5. crop_data/break",
    normal_data_dir: Optional[str] = None,
    output_dir: str = "6. train_data/sequences",
    sequence_length: int = 50,
    sort_by: str = 'height',
    normalize: bool = True,
    min_points: int = 5,
    use_cropped_normal: bool = True,
    use_sliding_window: bool = False,
    window_height: float = 0.3,
    window_degree: float = 90.0,
    stride_height: float = 0.1,
    stride_degree: float = 30.0,
    edit_data_dir: Optional[str] = None,
):
    """
    모든 크롭 파일과 정상 데이터를 처리하여 시퀀스 데이터 준비.
    
    Args:
        crop_base_dir: 크롭된 파일이 있는 기본 디렉토리 (슬라이딩 윈도우 모드에서는 무시)
        normal_data_dir: 정상 데이터 디렉토리 (None이면 use_cropped_normal에 따라 자동 설정)
        output_dir: 출력 디렉토리
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        normalize: 정규화 여부
        min_points: 최소 데이터 포인트 수 (이보다 적으면 제외)
        use_cropped_normal: 정상 데이터 크롭 파일 사용 여부
        use_sliding_window: 슬라이딩 윈도우 방식 사용 여부 (True면 평가 시와 동일한 방식)
        window_height: 윈도우 높이 범위 (슬라이딩 윈도우 모드)
        window_degree: 윈도우 각도 범위 (슬라이딩 윈도우 모드)
        stride_height: 높이 스트라이드 (슬라이딩 윈도우 모드)
        stride_degree: 각도 스트라이드 (슬라이딩 윈도우 모드)
        edit_data_dir: edit_pole_data 디렉토리 경로 (슬라이딩 윈도우 모드 사용 시)
    """
    # 슬라이딩 윈도우 모드
    if use_sliding_window:
        return process_with_sliding_window(
            edit_data_dir=edit_data_dir or "4. edit_pole_data",
            normal_data_dir=normal_data_dir or "4. edit_pole_data/normal",
            output_dir=output_dir,
            sequence_length=sequence_length,
            sort_by=sort_by,
            window_height=window_height,
            window_degree=window_degree,
            stride_height=stride_height,
            stride_degree=stride_degree
        )
    
    # 기존 방식 (크롭된 파일 사용)
    # 정상 데이터 디렉토리가 지정되지 않은 경우 기본값 사용
    if normal_data_dir is None and use_cropped_normal:
        normal_data_dir = "5. crop_data/normal"
    """
    모든 크롭 파일과 정상 데이터를 처리하여 시퀀스 데이터 준비.
    
    Args:
        crop_base_dir: 크롭된 파일이 있는 기본 디렉토리
        normal_data_dir: 정상 데이터 디렉토리 (선택사항)
        output_dir: 출력 디렉토리
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        normalize: 정규화 여부
        min_points: 최소 데이터 포인트 수 (이보다 적으면 제외)
        max_normal_samples_per_file: 정상 데이터 파일당 최대 샘플 수
    """
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 파단 데이터 파일 수집
    print("파단 데이터 파일 수집 중...")
    crop_files = collect_all_crop_files(crop_base_dir)
    
    # 정상 데이터 파일 수집
    normal_files = []
    if normal_data_dir:
        print(f"\n정상 데이터 파일 수집 중...")
        print(f"  정상 데이터 디렉토리: {normal_data_dir}")
        normal_files = collect_normal_data_files(normal_data_dir, use_cropped=use_cropped_normal)
        if normal_files:
            unique_files = len(set(f[0] for f in normal_files))
            print(f"  정상 데이터 파일: {unique_files}개")
            print(f"  정상 데이터 샘플: {len(normal_files)}개")
        else:
            print(f"  경고: 정상 데이터 파일을 찾을 수 없습니다.")
            print(f"  확인: 디렉토리가 올바른지, 파일이 존재하는지 확인하세요.")
    
    if not crop_files and not normal_files:
        print("처리할 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(crop_files)}개의 파단 파일, {len(normal_files)}개의 정상 샘플 발견")
    
    # 시퀀스 데이터 처리
    sequences = []
    labels = []
    metadata_list = []
    failed_files = []
    
    # 파단 데이터 처리
    if crop_files:
        print(f"\n파단 데이터 시퀀스 생성 중... (길이: {sequence_length}, 정렬 기준: {sort_by})")
        for csv_path, project_name, poleid, label in tqdm(crop_files, desc="파단 데이터 처리"):
            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sequence_length=sequence_length,
                sort_by=sort_by,
                normalize=normalize
            )
            
            if result is None:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'label': label,
                    'reason': '데이터 로드 실패 또는 형식 오류'
                })
                continue
            
            sequence, metadata = result
            
            # 최소 포인트 수 확인
            if metadata['original_length'] < min_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'label': label,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < {min_points})"
                })
                continue
            
            sequences.append(sequence)
            labels.append(label)  # 라벨 1: 파단
            metadata['csv_path'] = csv_path
            metadata['project_name'] = project_name
            metadata['poleid'] = poleid
            metadata_list.append(metadata)
    
    # 정상 데이터 처리
    if normal_files:
        print(f"\n정상 데이터 시퀀스 생성 중... (길이: {sequence_length}, 정렬 기준: {sort_by})")
        for csv_path, project_name, poleid, label in tqdm(normal_files, desc="정상 데이터 처리"):
            try:
                result = prepare_sequence_from_csv(
                    csv_path=csv_path,
                    sequence_length=sequence_length,
                    sort_by=sort_by,
                    normalize=normalize
                )
                
                if result is None:
                    failed_files.append({
                        'csv_path': csv_path,
                        'project_name': project_name,
                        'poleid': poleid,
                        'label': 0,
                        'reason': '데이터 로드 실패 또는 형식 오류'
                    })
                    continue
                
                sequence, metadata = result
                
                # 최소 포인트 수 확인
                if metadata['original_length'] < min_points:
                    failed_files.append({
                        'csv_path': csv_path,
                        'project_name': project_name,
                        'poleid': poleid,
                        'label': 0,
                        'reason': f"데이터 포인트 부족 ({metadata['original_length']} < {min_points})"
                    })
                    continue
                
                sequences.append(sequence)
                labels.append(label)  # 라벨 0: 정상
                metadata['csv_path'] = csv_path
                metadata['project_name'] = project_name
                metadata['poleid'] = poleid
                metadata_list.append(metadata)
            except Exception as e:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'label': 0,
                    'reason': f'데이터 처리 실패: {e}'
                })
                continue
    
    if not sequences:
        print("생성된 시퀀스 데이터가 없습니다.")
        return
    
    # 배열로 변환
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"\n시퀀스 데이터 생성 완료:")
    print(f"  총 샘플 수: {len(X)}")
    print(f"  시퀀스 형태: {X.shape}")
    print(f"  실패한 파일: {len(failed_files)}개")
    
    # 데이터 저장
    sequences_file = output_path / "break_sequences.npy"
    labels_file = output_path / "break_labels.npy"
    metadata_file = output_path / "break_sequences_metadata.json"
    
    np.save(sequences_file, X)
    np.save(labels_file, y)
    
    # 메타데이터 저장
    metadata_dict = {
        'total_samples': len(metadata_list),
        'sequence_length': sequence_length,
        'feature_names': ['x_value', 'y_value', 'z_value'],  # 각도와 높이는 학습 피처에서 제외
        'sort_by': sort_by,
        'normalize': normalize,
        'min_points': min_points,
        'samples': metadata_list,
        'failed_files': failed_files,
        'data_shape': list(X.shape),
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터 저장 완료:")
    print(f"  시퀀스 데이터: {sequences_file}")
    print(f"  라벨 데이터: {labels_file}")
    print(f"  메타데이터: {metadata_file}")
    
    # 통계 정보 출력
    break_samples = [i for i, label in enumerate(labels) if label == 1]
    normal_samples = [i for i, label in enumerate(labels) if label == 0]
    
    print(f"\n데이터 통계:")
    print(f"  총 샘플 수: {len(sequences)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  원본 데이터 포인트 수 범위: {min(m['original_length'] for m in metadata_list)} ~ {max(m['original_length'] for m in metadata_list)}")
    print(f"  평균 포인트 수: {np.mean([m['original_length'] for m in metadata_list]):.1f}")
    if metadata_list:
        print(f"  높이 범위: {min(m['height_min'] for m in metadata_list):.3f} ~ {max(m['height_max'] for m in metadata_list):.3f}m")
        print(f"  각도 범위: {min(m['degree_min'] for m in metadata_list):.1f} ~ {max(m['degree_max'] for m in metadata_list):.1f}°")


def main():
    parser = argparse.ArgumentParser(
        description="크롭된 파단 영역 데이터 또는 슬라이딩 윈도우 방식으로 시퀀스 데이터 생성하여 학습 데이터 준비"
    )
    parser.add_argument(
        "--crop-dir",
        type=str,
        default="5. crop_data/break",
        help="크롭된 파단 파일이 있는 기본 디렉토리 (상대 경로)",
    )
    parser.add_argument(
        "--normal-dir",
        type=str,
        default=None,
        help="정상 데이터 디렉토리 (상대 경로, 선택사항, 기본값: 5. crop_data/normal)",
    )
    parser.add_argument(
        "--normal-use-cropped",
        action="store_true",
        default=True,
        help="정상 데이터 크롭 파일 사용 (5. crop_data/normal/, 기본값: True)",
    )
    parser.add_argument(
        "--no-normal-cropped",
        action="store_false",
        dest="normal_use_cropped",
        help="정상 데이터 크롭 파일 사용 안 함 (processed 파일 사용)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="6. train_data/sequences",
        help="출력 디렉토리 (상대 경로)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="시퀀스 길이 (기본값: 50)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="height",
        choices=['height', 'degree'],
        help="정렬 기준 (기본값: height)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="정규화 사용 안 함",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="최소 데이터 포인트 수 (기본값: 5)",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        default=False,
        help="슬라이딩 윈도우 방식 사용 (평가 시와 동일한 방식, 기본값: False)",
    )
    parser.add_argument(
        "--edit-data-dir",
        type=str,
        default="4. edit_pole_data",
        help="edit_pole_data 디렉토리 경로 (슬라이딩 윈도우 모드 사용 시, 기본값: 4. edit_pole_data)",
    )
    parser.add_argument(
        "--window-height",
        type=float,
        default=0.3,
        help="윈도우 높이 범위 (슬라이딩 윈도우 모드, 기본값: 0.3m)",
    )
    parser.add_argument(
        "--window-degree",
        type=float,
        default=90.0,
        help="윈도우 각도 범위 (슬라이딩 윈도우 모드, 기본값: 90.0도)",
    )
    parser.add_argument(
        "--stride-height",
        type=float,
        default=0.1,
        help="높이 스트라이드 (슬라이딩 윈도우 모드, 기본값: 0.1m)",
    )
    parser.add_argument(
        "--stride-degree",
        type=float,
        default=30.0,
        help="각도 스트라이드 (슬라이딩 윈도우 모드, 기본값: 30.0도)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("파단 영역 시퀀스 학습 데이터 준비 시작")
    print("=" * 80)
    
    process_all_crops(
        crop_base_dir=args.crop_dir,
        normal_data_dir=args.normal_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        sort_by=args.sort_by,
        normalize=not args.no_normalize,
        min_points=args.min_points,
        use_cropped_normal=args.normal_use_cropped,
        use_sliding_window=args.sliding_window,
        window_height=args.window_height,
        window_degree=args.window_degree,
        stride_height=args.stride_height,
        stride_degree=args.stride_degree,
        edit_data_dir=args.edit_data_dir,
    )
    
    print("\n" + "=" * 80)
    print("시퀀스 데이터 준비 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
