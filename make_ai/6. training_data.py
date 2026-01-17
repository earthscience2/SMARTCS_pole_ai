#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4. merge_data에서 파단 및 정상 데이터를 2차원 그리드 데이터로 변환하여 학습 데이터를 준비하는 스크립트.

입력: 
  - 파단 데이터: 4. merge_data/break/ 디렉토리 (*_OUT_processed.csv 파일)
  - 정상 데이터: 4. merge_data/normal/ 디렉토리 (*_OUT_processed.csv 파일)
출력: 5. train_data/ 디렉토리에 2D 그리드 데이터 및 메타데이터 저장

2D 그리드 형식:
- 각 샘플: (height_bins, degree_bins, 3) 형태 (x, y, z 값)
- height 축: 높이 범위를 여러 구간으로 나눔 (기본: 50개 구간)
- degree 축: 각도 범위를 여러 구간으로 나눔 (기본: 72개 구간, 5도 간격)
- 각 (height, degree) 셀에 x, y, z 값 배치 (정규화된 값)
- 파단 데이터: 라벨 1, 파단 위치 정보 포함 (정규화된 break_height, break_degree)
- 정상 데이터: 라벨 0, 파단 위치 정보 없음 (break_positions는 [0.0, 0.0])

2D 그리드의 장점:
- 파단 위치 주변의 공간적 패턴을 더 잘 학습 가능
- height와 degree의 관계를 명확하게 표현
- 파단 위치 예측 정확도 향상
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d, griddata

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_crop_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    크롭된 CSV 파일 로드.
    
    Args:
        csv_path: CSV 파일 경로
    
    Returns:
        DataFrame 또는 None
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def prepare_2d_grid_from_csv(
    csv_path: str,
    height_bins: int = 50,
    degree_bins: int = 72,
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    CSV 파일에서 2D 그리드 데이터를 생성 (height × degree 그리드에 x, y, z 값 배치).
    
    Args:
        csv_path: CSV 파일 경로
        height_bins: 높이 축 구간 수 (기본: 50)
        degree_bins: 각도 축 구간 수 (기본: 72, 5도 간격)
        feature_min_max: 각 피처별 (min, max) 딕셔너리 (None이면 df에서 계산)
    
    Returns:
        tuple: (2D 그리드 배열 (height_bins, degree_bins, 3), 메타데이터) 또는 None
    """
    df = load_crop_csv(csv_path)
    if df is None:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    # 각 파일별 min/max 계산
    if feature_min_max is None:
        feature_min_max = {
            'height': (df['height'].min(), df['height'].max()),
            'degree': (df['degree'].min(), df['degree'].max()),
            'x_value': (df['x_value'].min(), df['x_value'].max()),
            'y_value': (df['y_value'].min(), df['y_value'].max()),
            'z_value': (df['z_value'].min(), df['z_value'].max()),
        }
    
    # 높이와 각도 범위
    h_min, h_max = feature_min_max['height']
    d_min, d_max = feature_min_max['degree']
    
    # 높이 범위가 0이거나 매우 작은 경우 처리
    if h_max <= h_min:
        h_max = h_min + 0.1  # 최소 범위 보장
    
    # 각도 범위가 0이거나 매우 작은 경우 처리
    if d_max <= d_min:
        d_max = d_min + 1.0  # 최소 범위 보장
    
    # 원본 데이터 포인트
    height_points = df['height'].values
    degree_points = df['degree'].values
    x_values = df['x_value'].values
    y_values = df['y_value'].values
    z_values = df['z_value'].values
    
    # x, y, z 값 정규화 (0~1)
    x_min, x_max = feature_min_max['x_value']
    y_min, y_max = feature_min_max['y_value']
    z_min, z_max = feature_min_max['z_value']
    
    if x_max > x_min:
        x_norm = (x_values - x_min) / (x_max - x_min)
    else:
        x_norm = np.zeros_like(x_values)
    
    if y_max > y_min:
        y_norm = (y_values - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y_values)
    
    if z_max > z_min:
        z_norm = (z_values - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z_values)
    
    # 2D 그리드 생성 (height × degree)
    height_grid = np.linspace(h_min, h_max, height_bins)
    degree_grid = np.linspace(d_min, d_max, degree_bins)
    height_mesh, degree_mesh = np.meshgrid(height_grid, degree_grid, indexing='ij')
    
    # 2D 그리드에 x, y, z 값 보간 (안전한 방법)
    # griddata를 사용하되, linear가 실패하면 nearest로 대체
    points = np.column_stack([height_points, degree_points])
    
    # 고유한 포인트 확인 (중복 제거)
    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    
    # 데이터 포인트가 충분한지 확인 (최소 3개 필요)
    if len(unique_points) < 3:
        # 데이터 포인트가 부족한 경우, nearest 방법 사용
        method_to_use = 'nearest'
    else:
        # degree 범위가 너무 작은지 확인
        degree_range = np.max(degree_points) - np.min(degree_points)
        height_range = np.max(height_points) - np.min(height_points)
        
        # 범위가 너무 작으면 nearest 사용
        if degree_range < 1e-6 or height_range < 1e-6:
            method_to_use = 'nearest'
        else:
            # linear 시도, 실패하면 nearest로 대체
            method_to_use = 'linear'
    
    # x 값 보간
    try:
        x_grid = griddata(
            points, x_norm,
            (height_mesh, degree_mesh),
            method=method_to_use,
            fill_value=0.0
        )
    except:
        # linear 실패 시 nearest로 대체
        x_grid = griddata(
            points, x_norm,
            (height_mesh, degree_mesh),
            method='nearest',
            fill_value=0.0
        )
    
    # y 값 보간
    try:
        y_grid = griddata(
            points, y_norm,
            (height_mesh, degree_mesh),
            method=method_to_use,
            fill_value=0.0
        )
    except:
        # linear 실패 시 nearest로 대체
        y_grid = griddata(
            points, y_norm,
            (height_mesh, degree_mesh),
            method='nearest',
            fill_value=0.0
        )
    
    # z 값 보간
    try:
        z_grid = griddata(
            points, z_norm,
            (height_mesh, degree_mesh),
            method=method_to_use,
            fill_value=0.0
        )
    except:
        # linear 실패 시 nearest로 대체
        z_grid = griddata(
            points, z_norm,
            (height_mesh, degree_mesh),
            method='nearest',
            fill_value=0.0
        )
    
    # 2D 그리드 배열 생성: (height_bins, degree_bins, 3)
    grid_data = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # NaN 값 처리 (보간되지 않은 영역)
    grid_data = np.nan_to_num(grid_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 메타데이터
    metadata = {
        'original_length': int(len(df)),
        'height_bins': int(height_bins),
        'degree_bins': int(degree_bins),
        'grid_shape': [int(height_bins), int(degree_bins), 3],
        'height_min': float(h_min),
        'height_max': float(h_max),
        'degree_min': float(d_min),
        'degree_max': float(d_max),
        'feature_min_max': {k: list(v) for k, v in feature_min_max.items()},
    }
    
    return grid_data, metadata


def collect_all_crop_files(merge_data_dir: str) -> List[Tuple[str, str, str, Path, int]]:
    """
    4. merge_data에서 파단 데이터 CSV 파일 수집.
    
    Args:
        merge_data_dir: merge_data 디렉토리 경로
    
    Returns:
        list: [(csv_path, project_name, poleid, pole_dir, label), ...] 리스트 (label=1)
    """
    break_base_dir = Path(current_dir) / merge_data_dir / "break"
    crop_files = []
    
    if not break_base_dir.exists():
        print(f"경고: 파단 데이터 디렉토리를 찾을 수 없습니다: {break_base_dir}")
        return crop_files
    
    # 모든 프로젝트 디렉토리 찾기
    for project_dir in break_base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        
        project_name = project_dir.name
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            
            poleid = pole_dir.name
            
            # *_OUT_processed.csv 파일 찾기
            for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                crop_files.append((str(csv_file), project_name, poleid, pole_dir, 1))  # 라벨 1: 파단
    
    return crop_files


def collect_normal_data_files(normal_data_dir: str) -> List[Tuple[str, str, str, int]]:
    """
    4. merge_data/normal에서 정상 데이터 파일 수집.
    
    Args:
        normal_data_dir: 정상 데이터 디렉토리
    
    Returns:
        list: [(csv_path, project_name, poleid, label), ...] 리스트 (label=0)
    """
    if not os.path.isabs(normal_data_dir):
        normal_path = Path(current_dir) / normal_data_dir
    else:
        normal_path = Path(normal_data_dir)
    
    normal_files = []
    
    if not normal_path.exists():
        print(f"경고: 정상 데이터 디렉토리를 찾을 수 없습니다: {normal_data_dir}")
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
            
            # *_OUT_processed.csv 또는 *_normal_processed.csv 파일 찾기
            for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                normal_files.append((str(csv_file), project_name, poleid, 0))  # 라벨 0: 정상
            for csv_file in pole_dir.glob("*_normal_processed.csv"):
                normal_files.append((str(csv_file), project_name, poleid, 0))  # 라벨 0: 정상
    
    return normal_files


def load_break_info_json(pole_dir: Path, csv_filename: str = None) -> Optional[Dict]:
    """
    break_info.json 로드.
    
    Args:
        pole_dir: 전주 디렉토리 경로
        csv_filename: CSV 파일명 (선택사항)
    
    Returns:
        break_info 딕셔너리 또는 None
    """
    # CSV 파일명이 제공된 경우, 해당 파일명과 매칭되는 break_info 파일 찾기
    if csv_filename:
        # 예: 0621R481_2_OUT_processed.csv -> 0621R481_2_OUT_processed_break_info.json
        base_name = csv_filename.replace('_OUT_processed.csv', '').replace('_normal_processed.csv', '')
        break_info_path = pole_dir / f"{base_name}_OUT_processed_break_info.json"
        
        if break_info_path.exists():
            try:
                with open(break_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
    
    # 기본 방식: {poleid}_break_info.json 찾기
    break_info_path = pole_dir / f"{pole_dir.name}_break_info.json"
    if break_info_path.exists():
        try:
            with open(break_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # 모든 break_info.json 파일 찾기 (폴백)
    break_info_files = list(pole_dir.glob("*_break_info.json"))
    if break_info_files:
        try:
            with open(break_info_files[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    return None


def process_cropped_data(
    merge_data_dir: str = "4. merge_data",
    normal_data_dir: str = "4. merge_data/normal",
    output_dir: str = "5. train_data",
    height_bins: int = 50,
    degree_bins: int = 72,
    min_points: int = 50,
):
    """
    4. merge_data에서 데이터를 처리하여 2D 그리드 데이터 준비.
    
    Args:
        merge_data_dir: merge_data 디렉토리 경로 (파단 데이터)
        normal_data_dir: 정상 데이터 디렉토리 (merge_data/normal)
        output_dir: 출력 디렉토리
        height_bins: 높이 축 구간 수 (기본: 50)
        degree_bins: 각도 축 구간 수 (기본: 72, 5도 간격)
        min_points: 최소 데이터 포인트 수
    """
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 파단 데이터 파일 수집
    print("파단 데이터 파일 수집 중...")
    crop_files = collect_all_crop_files(merge_data_dir)
    
    # 정상 데이터 파일 수집
    normal_files = []
    if normal_data_dir:
        print(f"\n정상 데이터 파일 수집 중...")
        normal_files = collect_normal_data_files(normal_data_dir)
        if normal_files:
            unique_files = len(set(f[0] for f in normal_files))
            print(f"  정상 데이터 파일: {unique_files}개")
            print(f"  정상 데이터 샘플: {len(normal_files)}개")
    
    if not crop_files and not normal_files:
        print("처리할 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(crop_files)}개의 파단 파일, {len(normal_files)}개의 정상 샘플 발견")
    print(f"\n각 파일별 min/max로 정규화합니다...")
    
    # 시퀀스 데이터 처리
    sequences = []
    labels = []
    break_positions = []  # 파단 위치 정보 (정규화된 값) [break_height, break_degree] 또는 None
    metadata_list = []
    failed_files = []
    
    # 파단 데이터 처리
    if crop_files:
        print(f"\n파단 데이터 2D 그리드 생성 중... (height_bins: {height_bins}, degree_bins: {degree_bins})")
        for csv_path, project_name, poleid, pole_dir, label in tqdm(crop_files, desc="파단 데이터 처리"):
            # break_info.json 로드
            csv_file_path = Path(csv_path)
            csv_filename = csv_file_path.name
            break_info = load_break_info_json(pole_dir, csv_filename=csv_filename)
            
            result = prepare_2d_grid_from_csv(
                csv_path=csv_path,
                height_bins=height_bins,
                degree_bins=degree_bins,
                feature_min_max=None  # 각 파일별 min/max 사용
            )
            
            if result is None:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '데이터 로드 실패 또는 형식 오류'
                })
                continue
            
            grid_data, metadata = result
            
            # 최소 포인트 수 확인 (50개 미만 제외)
            if metadata['original_length'] < 50:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < 50)"
                })
                continue
            
            # 추가 최소 포인트 수 확인 (min_points 파라미터)
            if metadata['original_length'] < min_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < {min_points})"
                })
                continue
            
            # 파단 위치 정보 추출 및 정규화
            normalized_break_height = None
            normalized_break_degree = None
            
            if break_info and break_info.get('breakstate') == 'B':
                break_height = break_info.get('breakheight')
                break_degree = break_info.get('breakdegree')
                
                if break_height is not None:
                    try:
                        break_height = float(break_height)
                        if break_height < 2.0:  # 2m 미만만 사용
                            h_min, h_max = metadata['height_min'], metadata['height_max']
                            if h_max > h_min:
                                normalized_break_height = (break_height - h_min) / (h_max - h_min)
                            else:
                                normalized_break_height = 0.0
                    except (ValueError, TypeError):
                        pass
                
                if break_degree is not None:
                    try:
                        break_degree = float(break_degree)
                        d_min, d_max = metadata['degree_min'], metadata['degree_max']
                        if d_max > d_min:
                            normalized_break_degree = (break_degree - d_min) / (d_max - d_min)
                        else:
                            normalized_break_degree = 0.0
                    except (ValueError, TypeError):
                        pass
            
            sequences.append(grid_data)
            labels.append(label)  # 라벨 1: 파단
            
            # 파단 위치 정보 저장 (정규화된 값)
            if normalized_break_height is not None and normalized_break_degree is not None:
                break_positions.append([normalized_break_height, normalized_break_degree])
            else:
                break_positions.append(None)
            
            metadata['csv_path'] = csv_path
            metadata['project_name'] = project_name
            metadata['poleid'] = poleid
            metadata['break_height_normalized'] = normalized_break_height
            metadata['break_degree_normalized'] = normalized_break_degree
            metadata_list.append(metadata)
    
    # 파단 샘플 수 계산
    break_sample_count = sum(1 for label in labels if label == 1)
    max_normal_samples = break_sample_count * 10  # 파단 샘플의 10배
    
    print(f"\n파단 샘플 수: {break_sample_count}개")
    print(f"정상 샘플 최대 수: {max_normal_samples}개 (파단 샘플의 10배)")
    
    # 정상 데이터 처리
    if normal_files:
        print(f"\n정상 데이터 2D 그리드 생성 중... (height_bins: {height_bins}, degree_bins: {degree_bins})")
        for csv_path, project_name, poleid, label in tqdm(normal_files, desc="정상 데이터 처리"):
            # 이미 충분한 샘플이 있으면 중단
            if len(sequences) - break_sample_count >= max_normal_samples:
                break
            
            result = prepare_2d_grid_from_csv(
                csv_path=csv_path,
                height_bins=height_bins,
                degree_bins=degree_bins,
                feature_min_max=None  # 각 파일별 min/max 사용
            )
            
            if result is None:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '데이터 로드 실패 또는 형식 오류'
                })
                continue
            
            grid_data, metadata = result
            
            # 최소 포인트 수 확인 (50개 미만 제외)
            if metadata['original_length'] < 50:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < 50)"
                })
                continue
            
            # 추가 최소 포인트 수 확인 (min_points 파라미터)
            if metadata['original_length'] < min_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < {min_points})"
                })
                continue
            
            # 정상 데이터 중 1000개 이상 포인트는 제외
            if metadata['original_length'] >= 1000:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"정상 데이터 포인트 수가 1000개 이상 ({metadata['original_length']} >= 1000)"
                })
                continue
            
            sequences.append(grid_data)
            labels.append(label)  # 라벨 0: 정상
            break_positions.append(None)  # 정상 데이터는 파단 위치 없음
            
            metadata['csv_path'] = csv_path
            metadata['project_name'] = project_name
            metadata['poleid'] = poleid
            metadata['break_height_normalized'] = None
            metadata['break_degree_normalized'] = None
            metadata_list.append(metadata)
            
            # 최대 샘플 수에 도달하면 중단
            if len(sequences) - break_sample_count >= max_normal_samples:
                break
        
        # 정상 샘플이 최대치를 초과하면 랜덤 샘플링
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        if len(normal_indices) > max_normal_samples:
            print(f"  생성된 정상 샘플: {len(normal_indices)}개 → {max_normal_samples}개로 샘플링")
            indices_to_keep = np.random.choice(normal_indices, size=max_normal_samples, replace=False)
            all_indices = [i for i, label in enumerate(labels) if label == 1] + list(indices_to_keep)
            sequences = [sequences[i] for i in all_indices]
            labels = [labels[i] for i in all_indices]
            break_positions = [break_positions[i] for i in all_indices]
            metadata_list = [metadata_list[i] for i in all_indices]
        
        print(f"  최종 정상 샘플 수: {len([l for l in labels if l == 0])}개")
    
    if not sequences:
        print("생성된 시퀀스 데이터가 없습니다.")
        return
    
    # 배열로 변환
    X = np.array(sequences)
    y = np.array(labels)
    
    # 파단 위치 정보를 배열로 변환 (파단 샘플만 유효한 값)
    break_positions_array = []
    for i, bp in enumerate(break_positions):
        if bp is not None:
            break_positions_array.append(bp)
        else:
            break_positions_array.append([0.0, 0.0])  # 정상 샘플은 [0, 0]
    break_positions_array = np.array(break_positions_array)
    
    print(f"\n2D 그리드 데이터 생성 완료:")
    print(f"  총 샘플 수: {len(X)}")
    print(f"  그리드 형태: {X.shape} (height_bins × degree_bins × features)")
    print(f"  파단 위치 정보 형태: {break_positions_array.shape}")
    print(f"  실패한 파일: {len(failed_files)}개")

    # 학습/테스트 데이터 분할 (7:3 비율)
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X,
        y,
        indices,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    # 파단 위치 정보도 동일한 인덱스로 분할
    bp_train = break_positions_array[train_indices]
    bp_test = break_positions_array[test_indices]

    print(f"\n데이터 분할 (학습:테스트 = 7:3 비율):")
    print(f"  학습 샘플 수: {len(X_train)}개")
    print(f"  테스트 샘플 수: {len(X_test)}개")

    # 저장 디렉토리 구성
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 학습/테스트 데이터 저장
    np.save(train_dir / "break_sequences_train.npy", X_train)
    np.save(train_dir / "break_labels_train.npy", y_train)
    np.save(train_dir / "break_positions_train.npy", bp_train)  # 파단 위치 정보
    
    np.save(test_dir / "break_sequences_test.npy", X_test)
    np.save(test_dir / "break_labels_test.npy", y_test)
    np.save(test_dir / "break_positions_test.npy", bp_test)  # 파단 위치 정보
    
    # 전체 데이터도 저장
    sequences_file = output_path / "break_sequences.npy"
    labels_file = output_path / "break_labels.npy"
    break_positions_file = output_path / "break_positions.npy"
    metadata_file = output_path / "break_sequences_metadata.json"
    
    np.save(sequences_file, X)
    np.save(labels_file, y)
    np.save(break_positions_file, break_positions_array)
    
    # 메타데이터 저장
    metadata_dict = {
        'total_samples': len(metadata_list),
        'data_type': '2d_grid',  # 2D 그리드 데이터임을 명시
        'height_bins': height_bins,
        'degree_bins': degree_bins,
        'grid_shape': [height_bins, degree_bins, 3],  # (height, degree, features)
        'feature_names': ['x_value', 'y_value', 'z_value'],  # 그리드에 배치된 피처
        'normalization_method': 'per_file',  # 각 파일별 min/max 사용
        'min_points': min_points,
        'samples': metadata_list,
        'failed_files': failed_files,
        'data_shape': list(X.shape),
        'break_positions_shape': list(break_positions_array.shape),
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터 저장 완료:")
    print(f"  학습 2D 그리드: {train_dir / 'break_sequences_train.npy'}")
    print(f"  학습 라벨: {train_dir / 'break_labels_train.npy'}")
    print(f"  학습 파단 위치: {train_dir / 'break_positions_train.npy'}")
    print(f"  테스트 2D 그리드: {test_dir / 'break_sequences_test.npy'}")
    print(f"  테스트 라벨: {test_dir / 'break_labels_test.npy'}")
    print(f"  테스트 파단 위치: {test_dir / 'break_positions_test.npy'}")
    print(f"  메타데이터: {metadata_file}")
    
    # 통계 정보 출력
    break_samples = [i for i, label in enumerate(labels) if label == 1]
    normal_samples = [i for i, label in enumerate(labels) if label == 0]
    
    print(f"\n데이터 통계:")
    print(f"  총 샘플 수: {len(sequences)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  파단 샘플 비율: {len(break_samples)/len(sequences)*100:.2f}%")


def main():
    print("=" * 60)
    print("2D 그리드 학습 데이터 준비 시작")
    print("=" * 60)
    
    process_cropped_data(
        merge_data_dir="4. merge_data",
        normal_data_dir="4. merge_data/normal",
        output_dir="5. train_data",
        height_bins=50,   # 높이 축 50개 구간
        degree_bins=72,   # 각도 축 72개 구간 (5도 간격)
        min_points=50,
    )
    
    print("\n" + "=" * 60)
    print("2D 그리드 데이터 준비 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
