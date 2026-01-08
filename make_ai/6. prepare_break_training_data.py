#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
크롭된 파단 영역 데이터에서 피처를 추출하여 학습 데이터를 준비하는 스크립트.

입력: raw_pole_data/crop_merged/ 디렉토리의 모든 크롭된 CSV 파일
출력: train_data/ 디렉토리에 피처 벡터 및 메타데이터 저장
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def extract_statistical_features(df: pd.DataFrame, axis: str) -> Dict[str, float]:
    """
    특정 축(x, y, z)에 대한 통계적 피처 추출.
    
    Args:
        df: 데이터프레임
        axis: 축 이름 ('x_value', 'y_value', 'z_value')
    
    Returns:
        dict: 통계 피처 딕셔너리
    """
    values = df[axis].values
    if len(values) == 0:
        return {}
    
    features = {
        f'{axis}_mean': float(np.mean(values)),
        f'{axis}_std': float(np.std(values)),
        f'{axis}_max': float(np.max(values)),
        f'{axis}_min': float(np.min(values)),
        f'{axis}_range': float(np.max(values) - np.min(values)),
        f'{axis}_median': float(np.median(values)),
        f'{axis}_q1': float(np.percentile(values, 25)),
        f'{axis}_q3': float(np.percentile(values, 75)),
        f'{axis}_iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
    }
    
    # 왜도와 첨도 계산 (scipy 없이 numpy로 계산)
    if len(values) > 1 and np.std(values) > 0:
        try:
            # 왜도 (skewness)
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                normalized = (values - mean) / std
                features[f'{axis}_skewness'] = float(np.mean(normalized ** 3))
                # 첨도 (kurtosis, excess kurtosis)
                features[f'{axis}_kurtosis'] = float(np.mean(normalized ** 4) - 3.0)
            else:
                features[f'{axis}_skewness'] = 0.0
                features[f'{axis}_kurtosis'] = 0.0
        except:
            features[f'{axis}_skewness'] = 0.0
            features[f'{axis}_kurtosis'] = 0.0
    else:
        features[f'{axis}_skewness'] = 0.0
        features[f'{axis}_kurtosis'] = 0.0
    
    # 변동계수 (coefficient of variation)
    if np.mean(values) != 0:
        features[f'{axis}_cv'] = float(np.std(values) / np.abs(np.mean(values)))
    else:
        features[f'{axis}_cv'] = 0.0
    
    return features


def extract_spatial_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    공간적 피처 추출 (높이, 각도 관련).
    
    Args:
        df: 데이터프레임
    
    Returns:
        dict: 공간 피처 딕셔너리
    """
    if df.empty:
        return {}
    
    height_values = df['height'].values
    degree_values = df['degree'].values
    
    features = {
        'height_range': float(np.max(height_values) - np.min(height_values)),
        'height_mean': float(np.mean(height_values)),
        'height_std': float(np.std(height_values)),
        'degree_range': float(np.max(degree_values) - np.min(degree_values)),
        'degree_mean': float(np.mean(degree_values)),
        'degree_std': float(np.std(degree_values)),
        'num_points': len(df),
        'height_density': len(df) / (np.max(height_values) - np.min(height_values) + 1e-6),
        'degree_density': len(df) / (np.max(degree_values) - np.min(degree_values) + 1e-6),
    }
    
    # 각도 범위가 360도를 넘어가는 경우 (wrap-around) 처리
    if features['degree_range'] > 180:
        features['degree_range'] = 360.0 - features['degree_range']
    
    return features


def extract_gradient_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    변화량 피처 추출 (gradient, 변화율).
    
    Args:
        df: 데이터프레임
    
    Returns:
        dict: 변화량 피처 딕셔너리
    """
    if len(df) < 2:
        return {}
    
    features = {}
    
    # 높이 방향으로 정렬
    df_sorted_h = df.sort_values('height')
    # 각도 방향으로 정렬
    df_sorted_d = df.sort_values('degree')
    
    for axis in ['x_value', 'y_value', 'z_value']:
        # 높이 방향 변화량
        if len(df_sorted_h) > 1:
            h_gradients = np.diff(df_sorted_h[axis].values)
            features[f'{axis}_height_grad_mean'] = float(np.mean(np.abs(h_gradients)))
            features[f'{axis}_height_grad_max'] = float(np.max(np.abs(h_gradients)))
            features[f'{axis}_height_grad_std'] = float(np.std(h_gradients))
        else:
            features[f'{axis}_height_grad_mean'] = 0.0
            features[f'{axis}_height_grad_max'] = 0.0
            features[f'{axis}_height_grad_std'] = 0.0
        
        # 각도 방향 변화량
        if len(df_sorted_d) > 1:
            d_gradients = np.diff(df_sorted_d[axis].values)
            features[f'{axis}_degree_grad_mean'] = float(np.mean(np.abs(d_gradients)))
            features[f'{axis}_degree_grad_max'] = float(np.max(np.abs(d_gradients)))
            features[f'{axis}_degree_grad_std'] = float(np.std(d_gradients))
        else:
            features[f'{axis}_degree_grad_mean'] = 0.0
            features[f'{axis}_degree_grad_max'] = 0.0
            features[f'{axis}_degree_grad_std'] = 0.0
        
        # 인접 포인트 간 차이 (전체)
        if len(df) > 1:
            # 모든 포인트 쌍의 차이 계산 (샘플링하여 계산량 감소)
            if len(df) > 100:
                # 데이터가 많으면 샘플링
                sample_df = df.sample(min(100, len(df)))
            else:
                sample_df = df
            
            values = sample_df[axis].values
            if len(values) > 1:
                diffs = []
                for i in range(len(values) - 1):
                    diffs.append(abs(values[i+1] - values[i]))
                features[f'{axis}_neighbor_diff_mean'] = float(np.mean(diffs))
                features[f'{axis}_neighbor_diff_max'] = float(np.max(diffs))
            else:
                features[f'{axis}_neighbor_diff_mean'] = 0.0
                features[f'{axis}_neighbor_diff_max'] = 0.0
        else:
            features[f'{axis}_neighbor_diff_mean'] = 0.0
            features[f'{axis}_neighbor_diff_max'] = 0.0
    
    return features


def extract_pattern_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    패턴 피처 추출 (상관관계 등).
    
    Args:
        df: 데이터프레임
    
    Returns:
        dict: 패턴 피처 딕셔너리
    """
    if len(df) < 2:
        return {}
    
    features = {}
    
    # x, y, z 축 간 상관관계
    axes = ['x_value', 'y_value', 'z_value']
    for i, axis1 in enumerate(axes):
        for axis2 in axes[i+1:]:
            corr = df[axis1].corr(df[axis2])
            if pd.notna(corr):
                features[f'{axis1}_{axis2}_corr'] = float(corr)
            else:
                features[f'{axis1}_{axis2}_corr'] = 0.0
    
    # 각 축의 변동성 (coefficient of variation)
    for axis in axes:
        values = df[axis].values
        if np.mean(values) != 0:
            features[f'{axis}_variability'] = float(np.std(values) / np.abs(np.mean(values)))
        else:
            features[f'{axis}_variability'] = 0.0
    
    return features


def extract_features_from_crop(csv_path: str) -> Tuple[Dict, Optional[str]]:
    """
    크롭된 CSV 파일에서 모든 피처를 추출.
    
    Args:
        csv_path: 크롭된 CSV 파일 경로
    
    Returns:
        tuple: (피처 딕셔너리, 오류 메시지 또는 None)
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None, "데이터가 비어 있습니다"
        
        # 필수 컬럼 확인
        required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"필수 컬럼이 없습니다: {missing_cols}"
        
        # NaN 값 제거
        df = df.dropna(subset=required_cols)
        if df.empty:
            return None, "유효한 데이터가 없습니다 (모두 NaN)"
        
        # 피처 추출
        features = {}
        
        # 통계적 피처 (각 축별)
        for axis in ['x_value', 'y_value', 'z_value']:
            features.update(extract_statistical_features(df, axis))
        
        # 공간적 피처
        features.update(extract_spatial_features(df))
        
        # 변화량 피처
        features.update(extract_gradient_features(df))
        
        # 패턴 피처
        features.update(extract_pattern_features(df))
        
        return features, None
        
    except Exception as e:
        return None, f"오류 발생: {str(e)}"


def collect_all_crop_files(crop_base_dir: str, file_pattern: str = "*_break_crop.csv") -> List[Tuple[str, str, str]]:
    """
    모든 크롭된 CSV 파일 경로를 수집.
    
    Args:
        crop_base_dir: 크롭된 파일이 있는 기본 디렉토리
        file_pattern: 파일명 패턴 (기본값: "*_break_crop.csv" 또는 "*_normal_crop.csv")
    
    Returns:
        list: [(csv_path, project_name, poleid), ...] 리스트
    """
    crop_path = Path(current_dir) / crop_base_dir
    
    if not crop_path.exists():
        print(f"경고: 크롭 디렉토리를 찾을 수 없습니다: {crop_base_dir}")
        return []
    
    crop_files = []
    
    # 모든 프로젝트 디렉토리 순회
    for project_dir in crop_path.iterdir():
        if not project_dir.is_dir():
            continue
        
        project_name = project_dir.name
        
        # 프로젝트 아래의 모든 전주 디렉토리 순회
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            
            poleid = pole_dir.name
            
            # 크롭된 CSV 파일 찾기 (패턴 매칭)
            for csv_file in pole_dir.glob(file_pattern):
                crop_files.append((str(csv_file), project_name, poleid))
    
    return crop_files


def process_all_crops(
    crop_base_dir: str = "raw_pole_data/crop_merged",
    output_dir: str = "train_data",
    min_points: int = 5,
    label: int = 1,
    file_pattern: str = "*_break_crop.csv",
    output_filename: str = "break_features.csv",
):
    """
    모든 크롭 파일을 처리하여 학습 데이터 준비.
    
    Args:
        crop_base_dir: 크롭된 파일이 있는 기본 디렉토리
        output_dir: 출력 디렉토리
        min_points: 최소 데이터 포인트 수 (이보다 적으면 제외)
        label: 라벨 값 (1: 파단, 0: 정상)
        file_pattern: 파일명 패턴
        output_filename: 출력 파일명
    """
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 크롭 파일 수집
    print("크롭된 파일 수집 중...")
    crop_files = collect_all_crop_files(crop_base_dir, file_pattern)
    
    if not crop_files:
        print("크롭된 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(crop_files)}개의 크롭 파일 발견")
    
    # 피처 추출
    all_features = []
    metadata = []
    failed_files = []
    
    print("\n피처 추출 중...")
    for csv_path, project_name, poleid in tqdm(crop_files, desc="처리 중"):
        features, error = extract_features_from_crop(csv_path)
        
        if features is None:
            failed_files.append({
                'file': csv_path,
                'project': project_name,
                'poleid': poleid,
                'error': error
            })
            continue
        
        # 최소 포인트 수 확인
        if features.get('num_points', 0) < min_points:
            failed_files.append({
                'file': csv_path,
                'project': project_name,
                'poleid': poleid,
                'error': f"데이터 포인트가 너무 적음 ({features.get('num_points', 0)} < {min_points})"
            })
            continue
        
        # 라벨 추가
        features['label'] = label
        features['poleid'] = poleid
        features['project_name'] = project_name
        features['csv_path'] = csv_path
        
        all_features.append(features)
        
        metadata.append({
            'poleid': poleid,
            'project_name': project_name,
            'csv_path': csv_path,
            'num_points': features.get('num_points', 0),
            'height_range': features.get('height_range', 0),
            'degree_range': features.get('degree_range', 0),
        })
    
    if not all_features:
        print("추출된 피처가 없습니다.")
        return
    
    # DataFrame 생성
    df_features = pd.DataFrame(all_features)
    
    # 피처 컬럼과 메타데이터 컬럼 분리
    feature_cols = [col for col in df_features.columns 
                    if col not in ['label', 'poleid', 'project_name', 'csv_path']]
    meta_cols = ['poleid', 'project_name', 'csv_path']
    
    # 컬럼 순서 재정렬 (피처 -> 라벨 -> 메타데이터)
    column_order = feature_cols + ['label'] + meta_cols
    df_features = df_features[column_order]
    
    # CSV 저장
    output_csv = output_path / output_filename
    df_features.to_csv(output_csv, index=False)
    print(f"\n피처 벡터 저장 완료: {output_csv}")
    print(f"  총 샘플 수: {len(df_features)}")
    print(f"  피처 수: {len(feature_cols)}")
    
    # 메타데이터 저장
    metadata_filename = output_filename.replace('.csv', '_metadata.json')
    output_metadata = output_path / metadata_filename
    with open(output_metadata, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(metadata),
            'samples': metadata,
            'failed_files': failed_files,
        }, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장 완료: {output_metadata}")
    
    # 피처 정보 저장
    feature_info = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'label': label,
        'feature_types': {
            'statistical': [col for col in feature_cols if any(axis in col for axis in ['x_value', 'y_value', 'z_value']) and any(stat in col for stat in ['mean', 'std', 'max', 'min', 'range', 'median', 'q1', 'q3', 'iqr', 'skewness', 'kurtosis', 'cv'])],
            'spatial': [col for col in feature_cols if col.startswith(('height_', 'degree_', 'num_'))],
            'gradient': [col for col in feature_cols if 'grad' in col or 'diff' in col],
            'pattern': [col for col in feature_cols if 'corr' in col or 'variability' in col],
        },
        'statistics': {
            col: {
                'mean': float(df_features[col].mean()),
                'std': float(df_features[col].std()),
                'min': float(df_features[col].min()),
                'max': float(df_features[col].max()),
            }
            for col in feature_cols
        }
    }
    
    info_filename = output_filename.replace('.csv', '_info.json')
    output_info = output_path / info_filename
    with open(output_info, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    print(f"피처 정보 저장 완료: {output_info}")
    
    # 실패한 파일 정보 출력
    if failed_files:
        print(f"\n경고: {len(failed_files)}개의 파일 처리 실패")
        print("자세한 내용은 break_metadata.json의 'failed_files' 섹션을 참조하세요.")


def main():
    parser = argparse.ArgumentParser(
        description="크롭된 영역 데이터에서 피처를 추출하여 학습 데이터를 준비"
    )
    parser.add_argument(
        "--crop-dir",
        type=str,
        default="raw_pole_data/crop_merged",
        help="크롭된 파일이 있는 기본 디렉토리 (상대 경로)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train_data",
        help="출력 디렉토리 (상대 경로)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="최소 데이터 포인트 수 (기본값: 5)",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        choices=[0, 1],
        help="라벨 값 (1: 파단, 0: 정상, 기본값: 1)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*_break_crop.csv",
        help="파일명 패턴 (기본값: *_break_crop.csv, 정상 데이터: *_normal_crop.csv)",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="출력 파일명 (기본값: label에 따라 자동 설정)",
    )

    args = parser.parse_args()
    
    # 출력 파일명 자동 설정
    if args.output_filename is None:
        if args.label == 1:
            args.output_filename = "break_features.csv"
        else:
            args.output_filename = "normal_features.csv"

    label_name = "파단" if args.label == 1 else "정상"
    print("=" * 80)
    print(f"{label_name} 영역 학습 데이터 준비 시작")
    print("=" * 80)
    
    process_all_crops(
        crop_base_dir=args.crop_dir,
        output_dir=args.output_dir,
        min_points=args.min_points,
        label=args.label,
        file_pattern=args.file_pattern,
        output_filename=args.output_filename,
    )
    
    print("\n" + "=" * 80)
    print("학습 데이터 준비 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()

