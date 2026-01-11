#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 LSTM 모델을 사용하여 새로운 데이터에서 파단을 검출하는 스크립트.

입력: 4. edit_pole_data/break/ 디렉토리의 processed CSV 파일들
      (각 파일은 특정 각도 범위를 담당: OUT_3(0-90도), OUT_4(90-180도), 
       OUT_5(180-270도), OUT_6(270-360도) 중 하나)

출력: 종합 검출 결과 보고서 (evaluation_report.md, evaluation_results.json)

검출 방식:
1. 슬라이딩 윈도우 방식으로 시퀀스 생성
   - 각 CSV 파일은 90도 단위의 각도 범위를 가지고 있음
   - 높이 0.3m, 각도 90도 윈도우로 슬라이딩
2. 각 윈도우에 대해 LSTM 모델로 예측
3. 임계값 기반으로 파단 위치 검출
4. 전체 검출 결과와 평가 결과를 하나의 보고서로 정리
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow.keras.models import load_model

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_lstm_model(model_path: str, info_path: Optional[str] = None) -> Tuple[tf.keras.Model, dict]:
    """
    학습된 LSTM 모델 로드.
    
    Args:
        model_path: 모델 파일 경로
        info_path: 모델 정보 파일 경로 (선택사항)
    
    Returns:
        tuple: (모델, 모델 정보)
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(model_path):
        model_path = os.path.join(current_dir, model_path)
    if info_path and not os.path.isabs(info_path):
        info_path = os.path.join(current_dir, info_path)
    
    model = load_model(model_path, compile=False)
    
    # 모델 컴파일 (예측을 위해)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    info = {}
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    else:
        # 기본값 설정
        info = {
            'sequence_length': 50,
            'n_features': 3,  # x, y, z만 사용 (각도와 높이는 제외)
            'sort_by': 'height',
        }
    
    return model, info


def prepare_sequence_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    sort_by: str = 'height',
    scaler: Optional[RobustScaler] = None
) -> Optional[np.ndarray]:
    """
    DataFrame에서 시퀀스 데이터 생성.
    
    Args:
        df: DataFrame (height, degree, x_value, y_value, z_value 포함)
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준 ('height' 또는 'degree')
        scaler: 스케일러 (None이면 정규화 안 함)
    
    Returns:
        시퀀스 배열 또는 None
    """
    if df.empty:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    # 정렬 (정렬에만 사용, 학습 피처에는 포함하지 않음)
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    
    # 피처 선택: [x, y, z] (각도와 높이는 파단과의 연관성이 없어 학습 피처에서 제외)
    features = df[['x_value', 'y_value', 'z_value']].values
    
    # 정규화
    if scaler is not None:
        features = scaler.transform(features)
    
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
    
    return sequence.reshape(1, sequence_length, -1)  # (1, seq_len, n_features)


def create_sliding_windows(
    df: pd.DataFrame,
    window_height: float,
    window_degree: float,
    stride_height: float,
    stride_degree: float,
    sequence_length: int,
    sort_by: str = 'height',
    scaler: Optional[RobustScaler] = None
) -> List[Tuple[np.ndarray, dict]]:
    """
    슬라이딩 윈도우 방식으로 시퀀스 생성.
    
    Args:
        df: DataFrame
        window_height: 윈도우 높이 범위
        window_degree: 윈도우 각도 범위
        stride_height: 높이 스트라이드
        stride_degree: 각도 스트라이드
        sequence_length: 시퀀스 길이
        sort_by: 정렬 기준
        scaler: 스케일러
    
    Returns:
        [(시퀀스, 메타데이터), ...] 리스트
    """
    sequences = []
    
    if df.empty:
        return sequences
    
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
                scaler=scaler
            )
            
            if sequence is not None:
                metadata = {
                    'height_start': float(h_start),
                    'height_end': float(h_end),
                    'degree_start': float(d_start),
                    'degree_end': float(d_end),
                    'center_height': float((h_start + h_end) / 2),
                    'center_degree': float((d_start + d_end) / 2),
                    'num_points': int(len(window_df))
                }
                sequences.append((sequence, metadata))
    
    return sequences


def detect_break_in_csv(
    csv_path: str,
    model: tf.keras.Model,
    model_info: dict,
    window_height: float = 0.3,
    window_degree: float = 90.0,
    stride_height: float = 0.1,
    stride_degree: float = 30.0,
    threshold: float = 0.5,
    use_sliding_window: bool = True,
    min_confidence: float = 0.6
) -> Optional[Dict]:
    """
    CSV 파일에서 파단 검출.
    
    Args:
        csv_path: CSV 파일 경로
        model: 학습된 LSTM 모델
        model_info: 모델 정보
        window_height: 윈도우 높이 범위
        window_degree: 윈도우 각도 범위
        stride_height: 높이 스트라이드
        stride_degree: 각도 스트라이드
        threshold: 분류 임계값
        use_sliding_window: 슬라이딩 윈도우 사용 여부
        min_confidence: 최소 신뢰도
    
    Returns:
        검출 결과 딕셔너리 또는 None
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  경고: CSV 로드 실패 ({csv_path}): {e}")
        return None
    
    if df.empty:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    sequence_length = model_info.get('sequence_length', 50)
    sort_by = model_info.get('sort_by', 'height')
    
    # 스케일러 생성 (모델 학습 시 사용한 것과 동일하게) - x, y, z만 사용
    scaler = RobustScaler()
    features = df[['x_value', 'y_value', 'z_value']].values
    scaler.fit(features)
    
    # 검출 방식 선택
    if use_sliding_window and len(df) > sequence_length:
        # 슬라이딩 윈도우 방식
        sequences_meta = create_sliding_windows(
            df,
            window_height=window_height,
            window_degree=window_degree,
            stride_height=stride_height,
            stride_degree=stride_degree,
            sequence_length=sequence_length,
            sort_by=sort_by,
            scaler=scaler
        )
        
        if not sequences_meta:
            return None
        
        # 예측
        sequences = np.vstack([seq for seq, _ in sequences_meta])
        predictions = model.predict(sequences, verbose=0).ravel()
        
        # 파단 검출 결과 추출
        detected_regions = []
        for (_, metadata), pred_prob in zip(sequences_meta, predictions):
            if pred_prob >= threshold and pred_prob >= min_confidence:
                detected_regions.append({
                    **metadata,
                    'confidence': float(pred_prob),
                })
        
        if not detected_regions:
            return {
                'break_detected': False,
                'confidence': 0.0,
                'regions': []
            }
        
        # 가장 높은 신뢰도를 가진 영역 선택
        best_region = max(detected_regions, key=lambda x: x['confidence'])
        
        # 파단 높이 정보 (예상 파단 높이)
        predicted_break_height = best_region['center_height']
        height_min = best_region['height_start']
        height_max = best_region['height_end']
        
        return {
            'break_detected': True,
            'confidence': best_region['confidence'],
            'detected_height': predicted_break_height,  # 예상 파단 높이 (중심값)
            'detected_degree': best_region['center_degree'],
            'height_range': [height_min, height_max],  # 파단 높이 범위
            'predicted_break_height': predicted_break_height,  # 예상 파단 높이 (명시적)
            'height_min': height_min,  # 파단 높이 최소값
            'height_max': height_max,  # 파단 높이 최대값
            'degree_range': [best_region['degree_start'], best_region['degree_end']],
            'regions': detected_regions[:5]  # 상위 5개만
        }
    
    else:
        # 전체 데이터를 하나의 시퀀스로
        sequence = prepare_sequence_from_dataframe(
            df,
            sequence_length=sequence_length,
            sort_by=sort_by,
            scaler=scaler
        )
        
        if sequence is None:
            return None
        
        # 예측
        prediction = model.predict(sequence, verbose=0)[0, 0]
        
        if prediction >= threshold and prediction >= min_confidence:
            h_center = df['height'].mean()
            h_min = df['height'].min()
            h_max = df['height'].max()
            d_center = df['degree'].mean()
            
            return {
                'break_detected': True,
                'confidence': float(prediction),
                'detected_height': float(h_center),  # 예상 파단 높이 (평균값)
                'predicted_break_height': float(h_center),  # 예상 파단 높이 (명시적)
                'height_min': float(h_min),  # 파단 높이 최소값
                'height_max': float(h_max),  # 파단 높이 최대값
                'detected_degree': float(d_center),
                'height_range': [float(h_min), float(h_max)],  # 파단 높이 범위
                'degree_range': [float(df['degree'].min()), float(df['degree'].max())],
                'regions': []
            }
        else:
            return {
                'break_detected': False,
                'confidence': float(prediction),
                'regions': []
            }


def load_break_info(pole_dir: Path, poleid: str) -> Optional[Dict]:
    """
    break_info.json 파일 로드.
    
    Args:
        pole_dir: 전주 디렉토리
        poleid: 전주 ID
    
    Returns:
        break_info 딕셔너리 또는 None
    """
    # 먼저 현재 디렉토리에서 찾기
    info_path = pole_dir / f"{poleid}_break_info.json"
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # edit_pole_data에서 찾기
    edit_info_path = Path(current_dir).parent / "4. edit_pole_data" / "break" / pole_dir.parent.name / poleid / f"{poleid}_break_info.json"
    if edit_info_path.exists():
        try:
            with open(edit_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # 현재 스크립트 디렉토리 기준으로 찾기
    edit_info_path2 = Path(current_dir) / "4. edit_pole_data" / "break" / pole_dir.parent.name / poleid / f"{poleid}_break_info.json"
    if edit_info_path2.exists():
        try:
            with open(edit_info_path2, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    return None


def check_csv_contains_break_location(
    csv_path: str,
    break_height: Optional[float],
    break_degree: Optional[float],
    height_margin: float = 0.15
) -> bool:
    """
    CSV 파일이 파단 위치를 포함하는지 확인.
    
    Args:
        csv_path: CSV 파일 경로
        break_height: 파단 높이 (None이면 확인 안 함)
        break_degree: 파단 각도 (None이면 확인 안 함)
        height_margin: 파단 높이 기준 ± margin (m)
    
    Returns:
        bool: 파단 위치를 포함하면 True
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'height' not in df.columns:
            return False
        
        # 높이 범위 확인
        if break_height is not None:
            h_min = break_height - height_margin
            h_max = break_height + height_margin
            
            file_h_min = df['height'].min()
            file_h_max = df['height'].max()
            
            # 높이 범위가 겹치는지 확인
            height_overlap = not (file_h_max < h_min or file_h_min > h_max)
            
            if not height_overlap:
                return False
        
        # 각도 범위 확인 (break_degree가 있는 경우)
        if break_degree is not None:
            if 'degree' not in df.columns:
                return False
            
            file_d_min = df['degree'].min()
            file_d_max = df['degree'].max()
            
            # 각도 범위 확인 (0~360도 wrap-around 고려)
            def is_degree_in_range(deg, d_min, d_max):
                # 범위가 0을 넘어가는 경우 (예: 350~10도)
                if d_min > d_max:
                    return deg >= d_min or deg <= d_max
                else:
                    return d_min <= deg <= d_max
            
            # break_degree가 파일의 각도 범위 안에 있는지 확인
            degree_contained = is_degree_in_range(break_degree, file_d_min, file_d_max)
            
            if not degree_contained:
                return False
        
        return True
        
    except Exception:
        return False


def evaluate_detection_results(
    detection_results: List[Dict],
    break_info: Optional[Dict],
    csv_filename: str,
    csv_path: Optional[str] = None
) -> Dict:
    """
    검출 결과 평가.
    
    Args:
        detection_results: 검출 결과 리스트
        break_info: 실제 파단 정보 (break_info.json)
        csv_filename: CSV 파일명
        csv_path: CSV 파일 전체 경로 (파단 위치 포함 여부 확인용)
    
    Returns:
        평가 결과 딕셔너리
    """
    # 실제 파단 정보 추출
    actual_break = False
    actual_height = None
    actual_degree = None
    
    if break_info:
        # break_info.json에 파단 정보가 있고 breakstate가 'B'인 경우
        if break_info.get('breakstate') == 'B':
            actual_height = break_info.get('breakheight')
            actual_degree = break_info.get('breakdegree')
            
            if actual_height is not None:
                try:
                    actual_height = float(actual_height)
                except (ValueError, TypeError):
                    actual_height = None
            
            if actual_degree is not None:
                try:
                    actual_degree = float(actual_degree)
                except (ValueError, TypeError):
                    actual_degree = None
            
            # CSV 파일이 실제로 파단 위치를 포함하는지 확인
            if csv_path and actual_height is not None:
                # 실제 높이가 2m 이상이면 제외
                if actual_height >= 2.0:
                    actual_break = False
                else:
                    # CSV 파일의 높이/각도 범위에 파단 위치가 포함되는지 확인
                    actual_break = check_csv_contains_break_location(
                        csv_path,
                        actual_height,
                        actual_degree,
                        height_margin=0.15
                    )
            elif actual_height is not None and actual_height < 2.0:
                # csv_path가 없으면 break_info.json만으로 판단 (기존 동작)
                actual_break = True
    
    # 검출 결과
    detected_break = False
    predicted_height = None
    
    if detection_results:
        # 가장 높은 신뢰도를 가진 검출 결과 사용
        best_detection = max(detection_results, key=lambda x: x.get('confidence', 0))
        detected_break = best_detection.get('break_detected', False)
        predicted_height = best_detection.get('predicted_break_height') or best_detection.get('detected_height')
    
    # 판단 정확도 계산
    if actual_break:
        if detected_break:
            result_type = 'TP'  # True Positive
        else:
            result_type = 'FN'  # False Negative
    else:
        if detected_break:
            result_type = 'FP'  # False Positive
        else:
            result_type = 'TN'  # True Negative
    
    # 높이 오차 계산 (파단이 실제로 있고 검출도 된 경우)
    height_error = None
    if actual_break and detected_break and actual_height is not None and predicted_height is not None:
        height_error = abs(predicted_height - actual_height)
    
    return {
        'actual_break': actual_break,
        'detected_break': detected_break,
        'result_type': result_type,
        'actual_height': actual_height,
        'predicted_height': predicted_height,
        'height_error': height_error
    }


def process_all_files(
    input_dir: str,
    model_path: str,
    model_info_path: Optional[str] = None,
    output_dir: str = "8. break_detection_results",
    window_height: float = 0.3,
    window_degree: float = 90.0,
    stride_height: float = 0.1,
    stride_degree: float = 30.0,
    threshold: float = 0.5,
    min_confidence: float = 0.6,
    use_sliding_window: bool = True,
    test_mode: bool = False,
    include_normal: bool = False,
    normal_sample_ratio: float = 0.1
):
    """
    모든 파일에 대해 파단 검출 수행.
    
    Args:
        input_dir: 입력 디렉토리 (프로젝트/전주ID 구조)
        model_path: 모델 파일 경로
        model_info_path: 모델 정보 파일 경로
        output_dir: 출력 디렉토리
        window_height: 윈도우 높이 범위
        window_degree: 윈도우 각도 범위
        stride_height: 높이 스트라이드
        stride_degree: 각도 스트라이드
        threshold: 분류 임계값
        min_confidence: 최소 신뢰도
        use_sliding_window: 슬라이딩 윈도우 사용 여부
    """
    # 모델 로드
    print("모델 로드 중...")
    model, model_info = load_lstm_model(model_path, model_info_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 입력 디렉토리 (break 데이터)
    input_path = Path(current_dir) / input_dir
    if not input_path.exists():
        print(f"오류: 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return
    
    # normal 데이터 디렉토리
    normal_input_path = None
    if include_normal:
        normal_input_dir = input_dir.replace('/break', '/normal')
        normal_input_path = Path(current_dir) / normal_input_dir
        if not normal_input_path.exists():
            print(f"경고: normal 데이터 디렉토리를 찾을 수 없습니다: {normal_input_dir}")
            include_normal = False
    
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다: {input_dir}")
        return
    
    total_processed = 0
    total_detected = 0
    
    # 평가 모드일 경우 평가 결과 저장
    evaluation_results = []
    
    print(f"\n파단 검출 시작...")
    print(f"입력 디렉토리 (break): {input_dir}")
    if include_normal and normal_input_path:
        print(f"입력 디렉토리 (normal): {normal_input_dir} (샘플링: {normal_sample_ratio*100:.0f}%)")
    print(f"출력 디렉토리: {output_dir}")
    print(f"임계값: {threshold}, 최소 신뢰도: {min_confidence}")
    if test_mode:
        print(f"평가 모드: ON (판단 정확도 및 높이 정확도 평가)")
    
    # 모든 검출 결과 수집
    all_detections = []
    
    for project_dir in projects:
        project_name = project_dir.name
        print(f"\n[프로젝트] {project_name}")
        
        # 프로젝트 아래의 모든 전주 디렉토리 찾기
        pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        for pole_dir in tqdm(pole_dirs, desc=f"  {project_name}"):
            poleid = pole_dir.name
            
            # break_info.json 로드 (평가 모드일 경우)
            break_info = None
            if test_mode:
                break_info = load_break_info(pole_dir, poleid)
            
            # CSV 파일 찾기 (processed 또는 merged)
            # 각 CSV 파일은 특정 각도 범위를 담당 (OUT_3: 0-90도, OUT_4: 90-180도, 
            # OUT_5: 180-270도, OUT_6: 270-360도)
            csv_files = list(pole_dir.glob("*_processed.csv")) + list(pole_dir.glob("*_merged.csv"))
            
            if not csv_files:
                continue
            
            # 각 CSV 파일에 대해 검출
            for csv_file in csv_files:
                result = detect_break_in_csv(
                    str(csv_file),
                    model,
                    model_info,
                    window_height=window_height,
                    window_degree=window_degree,
                    stride_height=stride_height,
                    stride_degree=stride_degree,
                    threshold=threshold,
                    min_confidence=min_confidence,
                    use_sliding_window=use_sliding_window
                )
                
                if result is None:
                    continue
                
                total_processed += 1
                
                if result['break_detected']:
                    total_detected += 1
                
                # 검출 결과 수집
                detection_record = {
                    'project_name': project_name,
                    'poleid': poleid,
                    'csv_file': csv_file.name,
                    'source_file': str(csv_file),
                    'data_type': 'break',
                    **result
                }
                all_detections.append(detection_record)
                
                # 평가 모드일 경우 평가 수행
                if test_mode:
                    eval_result = evaluate_detection_results(
                        [result],  # 검출 결과 리스트
                        break_info,
                        csv_file.name,
                        csv_path=str(csv_file)  # CSV 파일 경로 전달
                    )
                    
                    evaluation_results.append({
                        'project_name': project_name,
                        'poleid': poleid,
                        'csv_file': csv_file.name,
                        'data_type': 'break',
                        **eval_result
                    })
    
    # Normal 데이터 처리 (10% 샘플링)
    if include_normal and normal_input_path:
        print(f"\n" + "=" * 80)
        print(f"Normal 데이터 검출 시작 (샘플링: {normal_sample_ratio*100:.0f}%)")
        print(f"=" * 80)
        
        # normal 프로젝트 디렉토리 찾기
        normal_projects = [d for d in normal_input_path.iterdir() if d.is_dir()]
        
        # 모든 normal CSV 파일 수집
        all_normal_files = []
        for project_dir in normal_projects:
            project_name = project_dir.name
            pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
            for pole_dir in pole_dirs:
                poleid = pole_dir.name
                csv_files = list(pole_dir.glob("*_processed.csv")) + list(pole_dir.glob("*_merged.csv"))
                for csv_file in csv_files:
                    all_normal_files.append((project_dir, pole_dir, csv_file, project_name, poleid))
        
        # 10% 샘플링
        import random
        random.seed(42)  # 재현 가능성을 위해 시드 설정
        sample_size = max(1, int(len(all_normal_files) * normal_sample_ratio))
        sampled_normal_files = random.sample(all_normal_files, min(sample_size, len(all_normal_files)))
        
        print(f"전체 normal 파일: {len(all_normal_files)}개")
        print(f"샘플링된 파일: {len(sampled_normal_files)}개")
        
        for project_dir, pole_dir, csv_file, project_name, poleid in tqdm(sampled_normal_files, desc="  Normal 데이터"):
            result = detect_break_in_csv(
                str(csv_file),
                model,
                model_info,
                window_height=window_height,
                window_degree=window_degree,
                stride_height=stride_height,
                stride_degree=stride_degree,
                threshold=threshold,
                min_confidence=min_confidence,
                use_sliding_window=use_sliding_window
            )
            
            if result is None:
                continue
            
            total_processed += 1
            
            if result['break_detected']:
                total_detected += 1
            
            # 검출 결과 수집
            detection_record = {
                'project_name': project_name,
                'poleid': poleid,
                'csv_file': csv_file.name,
                'source_file': str(csv_file),
                'data_type': 'normal',
                **result
            }
            all_detections.append(detection_record)
            
            # 평가 모드일 경우 평가 수행 (normal 데이터는 실제 파단이 없으므로 FP 또는 TN)
            if test_mode:
                eval_result = evaluate_detection_results(
                    [result],
                    None,  # normal 데이터는 break_info.json이 없음
                    csv_file.name,
                    csv_path=str(csv_file)
                )
                evaluation_results.append({
                    'project_name': project_name,
                    'poleid': poleid,
                    'csv_file': csv_file.name,
                    'data_type': 'normal',
                    **eval_result
                })
    
    # 종합 보고서 생성 (검출 결과 + 평가 결과)
    create_comprehensive_report(
        all_detections=all_detections,
        evaluation_results=evaluation_results if test_mode else [],
        output_path=output_path,
        model_path=model_path,
        model_info=model_info,
        total_processed=total_processed,
        total_detected=total_detected,
        test_mode=test_mode
    )
    
    print(f"\n" + "=" * 80)
    print(f"파단 검출 완료")
    print(f"  처리된 파일: {total_processed}개")
    print(f"  검출된 파일: {total_detected}개")
    print(f"  검출률: {total_detected / total_processed * 100:.2f}%" if total_processed > 0 else "  검출률: 0%")
    print(f"=" * 80)


def create_comprehensive_report(
    all_detections: List[Dict],
    evaluation_results: List[Dict],
    output_path: Path,
    model_path: str,
    model_info: dict,
    total_processed: int,
    total_detected: int,
    test_mode: bool
):
    """
    종합 검출 결과 보고서 생성 (검출 결과 + 평가 결과).
    
    Args:
        all_detections: 모든 검출 결과
        evaluation_results: 평가 결과
        output_path: 출력 디렉토리
        model_path: 모델 경로
        model_info: 모델 정보
        total_processed: 처리된 파일 수
        total_detected: 검출된 파일 수
        test_mode: 평가 모드 여부
    """
    from datetime import datetime
    
    # 평가 결과가 있으면 평가 메트릭 계산
    if test_mode and evaluation_results:
        print_evaluation_results(evaluation_results, output_path, model_path, model_info)
        eval_summary = calculate_evaluation_summary(evaluation_results, model_path, model_info)
    else:
        eval_summary = None
    
    # 종합 보고서 생성
    report_lines = []
    report_lines.append("# 파단 검출 결과 종합 보고서\n")
    report_lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 모델 정보
    report_lines.append("## 1. 사용된 모델 정보\n")
    report_lines.append(f"- **모델 경로**: `{model_path}`\n")
    report_lines.append(f"- **모델 타입**: {model_info.get('model_type', 'unknown')}\n")
    report_lines.append(f"- **시퀀스 길이**: {model_info.get('sequence_length', 'unknown')}\n")
    report_lines.append(f"- **피처 수**: {model_info.get('n_features', 'unknown')}\n")
    report_lines.append(f"- **양방향 사용**: {model_info.get('use_bidirectional', 'unknown')}\n")
    report_lines.append(f"- **LSTM Units**: {model_info.get('lstm_units', 'unknown')}\n")
    report_lines.append(f"- **Dropout Rate**: {model_info.get('dropout_rate', 'unknown')}\n")
    report_lines.append(f"- **Learning Rate**: {model_info.get('learning_rate', 'unknown')}\n")
    report_lines.append(f"- **최적 임계값**: {model_info.get('optimal_threshold', 'unknown')}\n")
    
    # 검출 통계
    report_lines.append("\n## 2. 검출 통계\n")
    report_lines.append(f"- **처리된 파일 수**: {total_processed}개\n")
    report_lines.append(f"- **파단 검출된 파일 수**: {total_detected}개\n")
    report_lines.append(f"- **검출률**: {total_detected / total_processed * 100:.2f}%\n" if total_processed > 0 else "- **검출률**: 0%\n")
    
    # 검출된 파일 목록
    detected_files = [d for d in all_detections if d.get('break_detected', False)]
    if detected_files:
        report_lines.append(f"\n### 2.1 파단 검출된 파일 목록 (총 {len(detected_files)}개)\n")
        report_lines.append("| 프로젝트 | 전주ID | CSV 파일 | 신뢰도 | 예상 높이 (m) | 높이 범위 (m) |\n")
        report_lines.append("|----------|--------|----------|--------|---------------|---------------|\n")
        for det in detected_files[:50]:  # 상위 50개만
            height_range = det.get('height_range', [None, None])
            height_range_str = f"{height_range[0]:.3f}-{height_range[1]:.3f}" if height_range[0] is not None else "N/A"
            report_lines.append(f"| {det['project_name']} | {det['poleid']} | {det['csv_file']} | {det.get('confidence', 0):.4f} | {det.get('predicted_break_height', 'N/A'):.3f} | {height_range_str} |\n")
        if len(detected_files) > 50:
            report_lines.append(f"\n... 외 {len(detected_files) - 50}개\n")
    
    # 평가 결과 포함
    if eval_summary:
        report_lines.append("\n## 3. 평가 결과\n")
        
        # 판단 정확도
        cls_metrics = eval_summary.get('classification_metrics', {})
        if cls_metrics:
            report_lines.append("### 3.1 판단 정확도\n")
            report_lines.append("| 지표 | 값 | 비율 |\n")
            report_lines.append("|------|-----|------|\n")
            report_lines.append(f"| 정확도 (Accuracy) | {cls_metrics.get('accuracy', 0):.4f} | {cls_metrics.get('accuracy', 0)*100:.2f}% |\n")
            report_lines.append(f"| 정밀도 (Precision) | {cls_metrics.get('precision', 0):.4f} | {cls_metrics.get('precision', 0)*100:.2f}% |\n")
            report_lines.append(f"| 재현율 (Recall) | {cls_metrics.get('recall', 0):.4f} | {cls_metrics.get('recall', 0)*100:.2f}% |\n")
            report_lines.append(f"| F1-Score | {cls_metrics.get('f1_score', 0):.4f} | - |\n")
            
            cm = cls_metrics.get('confusion_matrix', {})
            if cm:
                report_lines.append("\n#### 혼동 행렬\n")
                report_lines.append("| | 실제 파단 | 실제 정상 | 합계 |\n")
                report_lines.append("|------|----------|----------|------|\n")
                tp, fp, fn, tn = cm.get('TP', 0), cm.get('FP', 0), cm.get('FN', 0), cm.get('TN', 0)
                total = tp + fp + fn + tn
                report_lines.append(f"| **검출 파단** | TP: {tp} | FP: {fp} | {tp + fp} |\n")
                report_lines.append(f"| **검출 정상** | FN: {fn} | TN: {tn} | {fn + tn} |\n")
                report_lines.append(f"| **합계** | {tp + fn} | {fp + tn} | {total} |\n")
        
        # 높이 정확도
        loc_metrics = eval_summary.get('location_metrics')
        if loc_metrics:
            report_lines.append("\n### 3.2 높이 정확도\n")
            report_lines.append(f"**평가 샘플 수 (TP)**: {loc_metrics.get('num_samples', 0)}개\n\n")
            report_lines.append("| 지표 | 값 (m) | 값 (cm) |\n")
            report_lines.append("|------|--------|----------|\n")
            report_lines.append(f"| 평균 오차 | {loc_metrics.get('mean_error', 0):.4f} | {loc_metrics.get('mean_error', 0)*100:.2f} |\n")
            report_lines.append(f"| 중앙값 오차 | {loc_metrics.get('median_error', 0):.4f} | {loc_metrics.get('median_error', 0)*100:.2f} |\n")
            report_lines.append(f"| RMSE | {loc_metrics.get('rmse', 0):.4f} | {loc_metrics.get('rmse', 0)*100:.2f} |\n")
            report_lines.append(f"| MAE | {loc_metrics.get('mae', 0):.4f} | {loc_metrics.get('mae', 0)*100:.2f} |\n")
    
    # 검출 결과 JSON 저장
    results_json = {
        'model_info': {
            'model_path': model_path,
            **{k: v for k, v in model_info.items() if k not in ['class_weight']}
        },
        'detection_summary': {
            'total_processed': total_processed,
            'total_detected': total_detected,
            'detection_rate': total_detected / total_processed if total_processed > 0 else 0
        },
        'detections': all_detections,
        'evaluation': eval_summary if eval_summary else None
    }
    
    results_json_file = output_path / "detection_results.json"
    with open(results_json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    # 보고서 저장
    report_file = output_path / "detection_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n종합 보고서 저장:")
    print(f"  보고서: {report_file}")
    print(f"  JSON: {results_json_file}")
    if eval_summary:
        print(f"  평가 결과: {output_path / 'evaluation_results.json'}")


def calculate_evaluation_summary(evaluation_results: List[Dict], model_path: str, model_info: dict) -> dict:
    """
    평가 결과 요약 계산 (print_evaluation_results에서 사용).
    """
    if not evaluation_results:
        return None
    
    y_true = [1 if r['actual_break'] else 0 for r in evaluation_results]
    y_pred = [1 if r['detected_break'] else 0 for r in evaluation_results]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    height_errors = [r['height_error'] for r in evaluation_results if r['height_error'] is not None]
    
    eval_summary = {
        'model_info': {
            'model_path': model_path,
            'model_type': model_info.get('model_type', 'unknown'),
            'sequence_length': model_info.get('sequence_length', 'unknown'),
            'n_features': model_info.get('n_features', 'unknown'),
            'optimal_threshold': model_info.get('optimal_threshold', 'unknown'),
            'use_bidirectional': model_info.get('use_bidirectional', 'unknown'),
            'lstm_units': model_info.get('lstm_units', 'unknown'),
            'dropout_rate': model_info.get('dropout_rate', 'unknown'),
            'learning_rate': model_info.get('learning_rate', 'unknown'),
        },
        'classification_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }
        },
        'location_metrics': None
    }
    
    if height_errors:
        height_errors_arr = np.array(height_errors)
        eval_summary['location_metrics'] = {
            'num_samples': len(height_errors),
            'mean_error': float(np.mean(height_errors_arr)),
            'median_error': float(np.median(height_errors_arr)),
            'std_error': float(np.std(height_errors_arr)),
            'rmse': float(np.sqrt(np.mean(height_errors_arr**2))),
            'mae': float(np.mean(np.abs(height_errors_arr))),
            'max_error': float(np.max(height_errors_arr)),
            'min_error': float(np.min(height_errors_arr))
        }
    
    return eval_summary


def print_evaluation_results(
    evaluation_results: List[Dict], 
    output_path: Path,
    model_path: str,
    model_info: dict
):
    """
    평가 결과 출력 및 저장.
    
    Args:
        evaluation_results: 평가 결과 리스트
        output_path: 출력 디렉토리
        model_path: 사용된 모델 경로
        model_info: 모델 정보
    """
    if not evaluation_results:
        return
    
    # 판단 정확도 계산
    y_true = [1 if r['actual_break'] else 0 for r in evaluation_results]
    y_pred = [1 if r['detected_break'] else 0 for r in evaluation_results]
    
    # 메트릭 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 높이 정확도 계산 (TP만)
    height_errors = [r['height_error'] for r in evaluation_results if r['height_error'] is not None]
    
    print(f"\n" + "=" * 80)
    print(f"판단 정확도 평가 결과")
    print(f"=" * 80)
    print(f"  정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  정밀도 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  재현율 (Recall): {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\n혼동 행렬:")
    print(f"  True Positive (TP): {tp}")
    print(f"  True Negative (TN): {tn}")
    print(f"  False Positive (FP): {fp}")
    print(f"  False Negative (FN): {fn}")
    
    if height_errors:
        height_errors_arr = np.array(height_errors)
        print(f"\n" + "=" * 80)
        print(f"높이 정확도 평가 결과 (TP 샘플: {len(height_errors)}개)")
        print(f"=" * 80)
        print(f"  평균 오차: {np.mean(height_errors_arr):.4f}m ({np.mean(height_errors_arr)*100:.2f}cm)")
        print(f"  중앙값 오차: {np.median(height_errors_arr):.4f}m ({np.median(height_errors_arr)*100:.2f}cm)")
        print(f"  표준편차: {np.std(height_errors_arr):.4f}m ({np.std(height_errors_arr)*100:.2f}cm)")
        print(f"  RMSE: {np.sqrt(np.mean(height_errors_arr**2)):.4f}m ({np.sqrt(np.mean(height_errors_arr**2))*100:.2f}cm)")
        print(f"  MAE: {np.mean(np.abs(height_errors_arr)):.4f}m ({np.mean(np.abs(height_errors_arr))*100:.2f}cm)")
        print(f"  최대 오차: {np.max(height_errors_arr):.4f}m ({np.max(height_errors_arr)*100:.2f}cm)")
        print(f"  최소 오차: {np.min(height_errors_arr):.4f}m ({np.min(height_errors_arr)*100:.2f}cm)")
    else:
        print(f"\n높이 정확도 평가: TP 샘플이 없어 계산할 수 없습니다.")
    
    # 평가 결과 JSON 저장
    eval_summary = {
        'model_info': {
            'model_path': model_path,
            'model_type': model_info.get('model_type', 'unknown'),
            'sequence_length': model_info.get('sequence_length', 'unknown'),
            'n_features': model_info.get('n_features', 'unknown'),
            'optimal_threshold': model_info.get('optimal_threshold', 'unknown'),
            'use_bidirectional': model_info.get('use_bidirectional', 'unknown'),
            'lstm_units': model_info.get('lstm_units', 'unknown'),
            'dropout_rate': model_info.get('dropout_rate', 'unknown'),
            'learning_rate': model_info.get('learning_rate', 'unknown'),
        },
        'classification_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }
        },
        'location_metrics': None
    }
    
    if height_errors:
        height_errors_arr = np.array(height_errors)
        eval_summary['location_metrics'] = {
            'num_samples': len(height_errors),
            'mean_error': float(np.mean(height_errors_arr)),
            'median_error': float(np.median(height_errors_arr)),
            'std_error': float(np.std(height_errors_arr)),
            'rmse': float(np.sqrt(np.mean(height_errors_arr**2))),
            'mae': float(np.mean(np.abs(height_errors_arr))),
            'max_error': float(np.max(height_errors_arr)),
            'min_error': float(np.min(height_errors_arr))
        }
    
    # JSON 저장
    eval_output_file = output_path / "evaluation_results.json"
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': eval_summary,
            'detailed_results': evaluation_results
        }, f, ensure_ascii=False, indent=2)
    
    # 마크다운 보고서 생성
    create_evaluation_report(eval_summary, evaluation_results, output_path, model_path, model_info)
    
    print(f"\n평가 결과 저장:")
    print(f"  JSON: {eval_output_file}")
    print(f"  보고서: {output_path / 'evaluation_report.md'}")
    print(f"=" * 80)


def create_evaluation_report(
    eval_summary: dict,
    evaluation_results: List[Dict],
    output_path: Path,
    model_path: str,
    model_info: dict
):
    """
    평가 결과 마크다운 보고서 생성.
    
    Args:
        eval_summary: 평가 요약 정보
        evaluation_results: 상세 평가 결과
        output_path: 출력 디렉토리
        model_path: 모델 경로
        model_info: 모델 정보
    """
    from datetime import datetime
    
    report_lines = []
    report_lines.append("# 파단 검출 모델 평가 보고서\n")
    report_lines.append(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 모델 정보
    report_lines.append("## 1. 사용된 모델 정보\n")
    report_lines.append(f"- **모델 경로**: `{model_path}`\n")
    report_lines.append(f"- **모델 타입**: {model_info.get('model_type', 'unknown')}\n")
    report_lines.append(f"- **시퀀스 길이**: {model_info.get('sequence_length', 'unknown')}\n")
    report_lines.append(f"- **피처 수**: {model_info.get('n_features', 'unknown')}\n")
    report_lines.append(f"- **양방향 사용**: {model_info.get('use_bidirectional', 'unknown')}\n")
    report_lines.append(f"- **LSTM Units**: {model_info.get('lstm_units', 'unknown')}\n")
    report_lines.append(f"- **Dropout Rate**: {model_info.get('dropout_rate', 'unknown')}\n")
    report_lines.append(f"- **Learning Rate**: {model_info.get('learning_rate', 'unknown')}\n")
    report_lines.append(f"- **최적 임계값**: {model_info.get('optimal_threshold', 'unknown')}\n")
    
    # 평가 데이터 정보
    total_samples = len(evaluation_results)
    actual_break_count = sum(1 for r in evaluation_results if r['actual_break'])
    detected_break_count = sum(1 for r in evaluation_results if r['detected_break'])
    
    report_lines.append("\n## 2. 평가 데이터 정보\n")
    report_lines.append(f"- **총 평가 샘플 수**: {total_samples}개\n")
    report_lines.append(f"- **실제 파단 샘플**: {actual_break_count}개 ({actual_break_count/total_samples*100:.2f}%)\n")
    report_lines.append(f"- **정상 샘플**: {total_samples - actual_break_count}개 ({(total_samples-actual_break_count)/total_samples*100:.2f}%)\n")
    report_lines.append(f"- **검출된 파단**: {detected_break_count}개\n")
    
    # 판단 정확도
    cls_metrics = eval_summary['classification_metrics']
    cm = cls_metrics['confusion_matrix']
    
    report_lines.append("\n## 3. 판단 정확도 평가 결과\n")
    report_lines.append("### 3.1 전체 성능 지표\n")
    report_lines.append("| 지표 | 값 | 비율 |\n")
    report_lines.append("|------|-----|------|\n")
    report_lines.append(f"| 정확도 (Accuracy) | {cls_metrics['accuracy']:.4f} | {cls_metrics['accuracy']*100:.2f}% |\n")
    report_lines.append(f"| 정밀도 (Precision) | {cls_metrics['precision']:.4f} | {cls_metrics['precision']*100:.2f}% |\n")
    report_lines.append(f"| 재현율 (Recall) | {cls_metrics['recall']:.4f} | {cls_metrics['recall']*100:.2f}% |\n")
    report_lines.append(f"| F1-Score | {cls_metrics['f1_score']:.4f} | - |\n")
    
    report_lines.append("\n### 3.2 혼동 행렬 (Confusion Matrix)\n")
    report_lines.append("| | 실제 파단 | 실제 정상 | 합계 |\n")
    report_lines.append("|------|----------|----------|------|\n")
    report_lines.append(f"| **검출 파단** | TP: {cm['TP']} | FP: {cm['FP']} | {cm['TP'] + cm['FP']} |\n")
    report_lines.append(f"| **검출 정상** | FN: {cm['FN']} | TN: {cm['TN']} | {cm['FN'] + cm['TN']} |\n")
    report_lines.append(f"| **합계** | {cm['TP'] + cm['FN']} | {cm['FP'] + cm['TN']} | {total_samples} |\n")
    
    # 높이 정확도
    if eval_summary['location_metrics']:
        loc_metrics = eval_summary['location_metrics']
        report_lines.append("\n## 4. 높이 정확도 평가 결과\n")
        report_lines.append(f"**평가 샘플 수 (TP)**: {loc_metrics['num_samples']}개\n\n")
        report_lines.append("### 4.1 높이 오차 통계\n")
        report_lines.append("| 지표 | 값 (m) | 값 (cm) |\n")
        report_lines.append("|------|--------|----------|\n")
        report_lines.append(f"| 평균 오차 (Mean Error) | {loc_metrics['mean_error']:.4f} | {loc_metrics['mean_error']*100:.2f} |\n")
        report_lines.append(f"| 중앙값 오차 (Median Error) | {loc_metrics['median_error']:.4f} | {loc_metrics['median_error']*100:.2f} |\n")
        report_lines.append(f"| 표준편차 (Std Error) | {loc_metrics['std_error']:.4f} | {loc_metrics['std_error']*100:.2f} |\n")
        report_lines.append(f"| RMSE | {loc_metrics['rmse']:.4f} | {loc_metrics['rmse']*100:.2f} |\n")
        report_lines.append(f"| MAE | {loc_metrics['mae']:.4f} | {loc_metrics['mae']*100:.2f} |\n")
        report_lines.append(f"| 최대 오차 | {loc_metrics['max_error']:.4f} | {loc_metrics['max_error']*100:.2f} |\n")
        report_lines.append(f"| 최소 오차 | {loc_metrics['min_error']:.4f} | {loc_metrics['min_error']*100:.2f} |\n")
        
        # 높이 오차가 큰 사례 분석
        tp_cases = [r for r in evaluation_results if r.get('result_type') == 'TP' and r.get('height_error') is not None]
        if tp_cases:
            # 높이 오차 기준으로 정렬 (내림차순)
            tp_cases_sorted = sorted(tp_cases, key=lambda x: x.get('height_error', 0), reverse=True)
            
            # 높이 오차가 큰 상위 사례 (1m 이상 또는 상위 20개)
            large_error_cases = [r for r in tp_cases_sorted if r.get('height_error', 0) >= 1.0]
            if len(large_error_cases) > 20:
                large_error_cases = large_error_cases[:20]
            elif len(tp_cases_sorted) > 20:
                large_error_cases = tp_cases_sorted[:20]
            
            report_lines.append("\n### 4.2 높이 오차가 큰 사례 (상위 20개 또는 1m 이상)\n")
            report_lines.append("| 순위 | 프로젝트 | 전주ID | CSV 파일 | 실제 높이 (m) | 예측 높이 (m) | 오차 (m) | 오차 (cm) |\n")
            report_lines.append("|------|----------|--------|----------|---------------|---------------|----------|-----------|\n")
            
            for idx, case in enumerate(large_error_cases, 1):
                actual_h = case.get('actual_height', 'N/A')
                predicted_h = case.get('predicted_height', 'N/A')
                error = case.get('height_error', 0)
                
                if isinstance(actual_h, (int, float)):
                    actual_h_str = f"{actual_h:.3f}"
                else:
                    actual_h_str = str(actual_h)
                
                if isinstance(predicted_h, (int, float)):
                    predicted_h_str = f"{predicted_h:.3f}"
                else:
                    predicted_h_str = str(predicted_h)
                
                error_cm = error * 100 if isinstance(error, (int, float)) else 0
                
                report_lines.append(f"| {idx} | {case['project_name']} | {case['poleid']} | {case['csv_file']} | {actual_h_str} | {predicted_h_str} | {error:.3f} | {error_cm:.2f} |\n")
            
            # 오차 분포 분석
            if len(tp_cases_sorted) > 0:
                errors = [r.get('height_error', 0) for r in tp_cases_sorted]
                error_1m_above = sum(1 for e in errors if e >= 1.0)
                error_50cm_above = sum(1 for e in errors if e >= 0.5)
                error_10cm_below = sum(1 for e in errors if e <= 0.1)
                
                report_lines.append("\n#### 오차 분포 분석\n")
                report_lines.append(f"- **1m 이상 오차**: {error_1m_above}개 ({error_1m_above/len(errors)*100:.2f}%)\n")
                report_lines.append(f"- **50cm 이상 오차**: {error_50cm_above}개 ({error_50cm_above/len(errors)*100:.2f}%)\n")
                report_lines.append(f"- **10cm 이하 오차**: {error_10cm_below}개 ({error_10cm_below/len(errors)*100:.2f}%)\n")
    else:
        report_lines.append("\n## 4. 높이 정확도 평가 결과\n")
        report_lines.append("**평가 불가**: True Positive 샘플이 없어 높이 정확도를 계산할 수 없습니다.\n")
    
    # 오류 사례 분석
    report_lines.append("\n## 5. 오류 사례 분석\n")
    
    # False Positive
    fp_cases = [r for r in evaluation_results if r['result_type'] == 'FP']
    report_lines.append(f"### 5.1 False Positive (FP) - {len(fp_cases)}개\n")
    report_lines.append("정상인데 파단으로 잘못 검출한 경우\n\n")
    if fp_cases:
        report_lines.append("| 프로젝트 | 전주ID | CSV 파일 |\n")
        report_lines.append("|----------|--------|----------|\n")
        for case in fp_cases[:10]:  # 상위 10개만
            report_lines.append(f"| {case['project_name']} | {case['poleid']} | {case['csv_file']} |\n")
        if len(fp_cases) > 10:
            report_lines.append(f"\n... 외 {len(fp_cases) - 10}개\n")
    else:
        report_lines.append("없음\n")
    
    # False Negative
    fn_cases = [r for r in evaluation_results if r['result_type'] == 'FN']
    report_lines.append(f"\n### 5.2 False Negative (FN) - {len(fn_cases)}개\n")
    report_lines.append("파단인데 검출하지 못한 경우\n\n")
    if fn_cases:
        report_lines.append("| 프로젝트 | 전주ID | CSV 파일 | 실제 높이 (m) |\n")
        report_lines.append("|----------|--------|----------|---------------|\n")
        for case in fn_cases[:10]:  # 상위 10개만
            actual_h = case.get('actual_height', 'N/A')
            if isinstance(actual_h, float):
                actual_h = f"{actual_h:.3f}"
            report_lines.append(f"| {case['project_name']} | {case['poleid']} | {case['csv_file']} | {actual_h} |\n")
        if len(fn_cases) > 10:
            report_lines.append(f"\n... 외 {len(fn_cases) - 10}개\n")
    else:
        report_lines.append("없음\n")
    
    # 요약
    report_lines.append("\n## 6. 평가 요약\n")
    report_lines.append(f"- **전체 정확도**: {cls_metrics['accuracy']*100:.2f}%\n")
    report_lines.append(f"- **정밀도**: {cls_metrics['precision']*100:.2f}% (검출된 파단 중 실제 파단 비율)\n")
    report_lines.append(f"- **재현율**: {cls_metrics['recall']*100:.2f}% (실제 파단 중 검출된 비율)\n")
    if eval_summary['location_metrics']:
        report_lines.append(f"- **높이 평균 오차**: {eval_summary['location_metrics']['mean_error']*100:.2f}cm\n")
        report_lines.append(f"- **높이 RMSE**: {eval_summary['location_metrics']['rmse']*100:.2f}cm\n")
    
    # 성능 분석 및 개선 제안
    report_lines.append("\n## 7. 성능 분석 및 개선 제안\n")
    
    # 문제점 분석
    fp_count = cm.get('FP', 0)
    tn_count = cm.get('TN', 0)
    total_normal = fp_count + tn_count
    fp_rate = (fp_count / total_normal * 100) if total_normal > 0 else 0
    tp_count = cm.get('TP', 0)
    
    # 학습 시 성능과 비교
    val_precision = model_info.get('val_precision', None)
    val_recall = model_info.get('val_recall', None)
    
    report_lines.append("### 7.1 현재 문제점\n")
    report_lines.append(f"- **정밀도 (Precision)**: {cls_metrics['precision']*100:.2f}% - **매우 낮음** ⚠️\n")
    if val_precision:
        report_lines.append(f"  - 학습 시 검증 정밀도: {val_precision*100:.2f}% (현재보다 {abs(val_precision - cls_metrics['precision'])*100:.2f}%p 높음)\n")
    report_lines.append(f"- **False Positive 비율**: {fp_rate:.2f}% (정상 데이터 중 파단으로 잘못 검출)\n")
    report_lines.append(f"  - FP: {fp_count}개 / 정상 총 {total_normal}개\n")
    report_lines.append(f"- **재현율 (Recall)**: {cls_metrics['recall']*100:.2f}% - 매우 높음 ✓\n")
    if val_recall:
        report_lines.append(f"  - 학습 시 검증 재현율: {val_recall*100:.2f}%\n")
    report_lines.append(f"- **높이 평균 오차**: {eval_summary['location_metrics']['mean_error']*100:.2f}cm (학습 시: {model_info.get('val_location_accuracy', {}).get('mean_error', 0)*100 if model_info.get('val_location_accuracy') else 0:.2f}cm)\n")
    
    if cls_metrics['precision'] < 0.3:
        report_lines.append("\n⚠️ **심각**: 정밀도가 30% 미만입니다. 검출된 파단 중 70% 이상이 오검출입니다.\n")
        report_lines.append("이 모델은 실용적으로 사용하기 어려울 수 있습니다.\n")
    elif cls_metrics['precision'] < 0.5:
        report_lines.append("\n⚠️ **주의**: 정밀도가 50% 미만입니다. 검출된 파단 중 절반 이상이 오검출입니다.\n")
    
    # 원인 분석
    report_lines.append("\n### 7.2 원인 분석\n")
    report_lines.append("#### 학습 데이터 vs 평가 데이터 차이\n")
    report_lines.append("- **학습 시**: 크롭된 데이터 (파단 ±0.15m, 정상 중간 높이 ±0.15m)\n")
    report_lines.append("- **평가 시**: 전체 CSV 파일 (높이 0~2m, 각도 90도 범위)\n")
    report_lines.append("- 평가 데이터가 더 넓은 범위를 포함하여 오검출이 증가할 수 있음\n")
    
    report_lines.append("\n#### 검출 방식 차이\n")
    report_lines.append("- **학습 시**: 크롭된 특정 영역만 사용\n")
    report_lines.append("- **평가 시**: 슬라이딩 윈도우로 전체 범위를 검사\n")
    report_lines.append("- 슬라이딩 윈도우 방식으로 인해 정상 영역도 파단으로 오검출될 수 있음\n")
    
    report_lines.append("\n#### 데이터 분포\n")
    report_lines.append(f"- **평가 데이터**: 파단 {actual_break_count}개 ({actual_break_count/total_samples*100:.2f}%), 정상 {total_normal}개 ({total_normal/total_samples*100:.2f}%)\n")
    report_lines.append("- 정상 데이터가 훨씬 많아서 정밀도에 큰 영향을 미침\n")
    
    # 개선 제안
    report_lines.append("\n### 7.3 개선 제안\n")
    report_lines.append("#### 1. 임계값 상향 조정 (가장 빠른 개선 방법)\n")
    report_lines.append(f"- **현재 임계값**: {model_info.get('optimal_threshold', 'unknown')}\n")
    report_lines.append("- **권장**: 임계값을 높여 False Positive를 줄이기\n")
    report_lines.append("  ```bash\n")
    report_lines.append(f"  python make_ai/8. detect_break_with_lstm.py --threshold 0.85\n")
    report_lines.append(f"  python make_ai/8. detect_break_with_lstm.py --threshold 0.90\n")
    report_lines.append("  ```\n")
    report_lines.append("- **예상 효과**: 정밀도 향상, 재현율 약간 감소\n")
    
    report_lines.append("\n#### 2. 최소 신뢰도 상향 조정\n")
    report_lines.append("- **현재 최소 신뢰도**: 0.6 (기본값)\n")
    report_lines.append("- **권장**: 0.7 이상으로 상향\n")
    report_lines.append("  ```bash\n")
    report_lines.append(f"  python make_ai/8. detect_break_with_lstm.py --min-confidence 0.75\n")
    report_lines.append("  ```\n")
    
    report_lines.append("\n#### 3. 다른 모델 테스트\n")
    report_lines.append("- `model_002`, `model_003` 등 다른 모델도 테스트하여 더 나은 성능을 가진 모델 찾기\n")
    report_lines.append("  ```bash\n")
    report_lines.append("  python make_ai/8. detect_break_with_lstm.py --model 7. models_ensemble/model_002_.../model.keras\n")
    report_lines.append("  ```\n")
    
    report_lines.append("\n#### 4. 모델 재학습 고려\n")
    report_lines.append("- 평가 데이터 분포에 맞게 더 많은 정상 샘플로 재학습\n")
    report_lines.append("- 또는 학습 시 평가와 동일한 방식(슬라이딩 윈도우)으로 데이터 준비\n")
    
    report_lines.append("\n#### 5. 검출 후처리 추가\n")
    report_lines.append("- 검출된 영역의 신뢰도 분포를 확인하여 높은 신뢰도만 선택\n")
    report_lines.append("- 연속된 영역에서 가장 신뢰도가 높은 영역만 선택\n")
    
    # 높이 오차 분석
    if eval_summary['location_metrics']:
        loc_metrics = eval_summary['location_metrics']
        tp_cases = [r for r in evaluation_results if r.get('result_type') == 'TP' and r.get('height_error') is not None]
        if tp_cases:
            errors = [r.get('height_error', 0) for r in tp_cases]
            error_50cm_above = sum(1 for e in errors if e >= 0.5)
            error_1m_above = sum(1 for e in errors if e >= 1.0)
            
            report_lines.append("\n#### 6. 높이 오차 개선\n")
            report_lines.append(f"- **50cm 이상 오차**: {error_50cm_above}개 ({error_50cm_above/len(errors)*100:.2f}%)\n")
            report_lines.append(f"- **1m 이상 오차**: {error_1m_above}개 ({error_1m_above/len(errors)*100:.2f}%)\n")
            if error_50cm_above / len(errors) > 0.4:
                report_lines.append("- 높이 오차가 큰 사례가 많음 → 슬라이딩 윈도우 크기 또는 스트라이드 조정 고려\n")
    
    # 보고서 저장
    report_file = output_path / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="학습된 LSTM 모델을 사용하여 파단 검출"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="7. models/break_pattern_lstm_best.keras",
        help="학습된 모델 파일 경로 (기본값: 7. models/break_pattern_lstm_best.keras)",
    )
    parser.add_argument(
        "--model-info",
        type=str,
        default=None,
        help="모델 정보 파일 경로 (기본값: 모델 디렉토리의 model_info.json)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="4. edit_pole_data/break",
        help="입력 디렉토리 (프로젝트/전주ID 구조, break 데이터)",
    )
    parser.add_argument(
        "--no-include-normal",
        action="store_false",
        dest="include_normal",
        default=True,
        help="Normal 데이터 제외 (기본값: Normal 데이터 포함, 10% 샘플링)",
    )
    parser.add_argument(
        "--normal-sample-ratio",
        type=float,
        default=0.1,
        help="Normal 데이터 샘플링 비율 (기본값: 0.1, 10%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="8. break_detection_results",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--window-height",
        type=float,
        default=0.3,
        help="윈도우 높이 범위 (m, 기본값: 0.3)",
    )
    parser.add_argument(
        "--window-degree",
        type=float,
        default=90.0,
        help="윈도우 각도 범위 (deg, 기본값: 90.0)",
    )
    parser.add_argument(
        "--stride-height",
        type=float,
        default=0.1,
        help="높이 스트라이드 (m, 기본값: 0.1)",
    )
    parser.add_argument(
        "--stride-degree",
        type=float,
        default=30.0,
        help="각도 스트라이드 (deg, 기본값: 30.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="분류 임계값 (기본값: 0.5)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="최소 신뢰도 (기본값: 0.6)",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="슬라이딩 윈도우 사용 안 함",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="단일 CSV 파일 경로 (단일 파일 처리용)",
    )
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        default=False,
        help="평가 모드 비활성화 (평가 없이 검출만 수행, 기본값: 평가 모드 ON)",
    )

    args = parser.parse_args()
    
    # 평가 모드 설정: --no-evaluation이 False면 평가 모드 ON
    test_mode = not args.no_evaluation
    
    print(f"\n평가 모드: {'ON' if test_mode else 'OFF'}")
    
    # 모델 정보 경로 자동 설정
    if args.model_info is None:
        model_path_obj = Path(args.model)
        if not model_path_obj.is_absolute():
            model_path_obj = Path(current_dir) / args.model
        
        # 모델 디렉토리에서 model_info.json 찾기
        model_dir = model_path_obj.parent
        info_file = model_dir / "model_info.json"
        if info_file.exists():
            args.model_info = str(info_file)
        else:
            # 기본 위치에서 찾기
            default_info = Path(current_dir) / "7. models" / "break_pattern_lstm_info.json"
            if default_info.exists():
                args.model_info = str(default_info)

    # 모델 로드
    model, model_info = load_lstm_model(args.model, args.model_info)

    # 단일 파일 처리 모드
    if args.csv_file is not None:
        result = detect_break_in_csv(
            args.csv_file,
            model,
            model_info,
            window_height=args.window_height,
            window_degree=args.window_degree,
            stride_height=args.stride_height,
            stride_degree=args.stride_degree,
            threshold=args.threshold,
            min_confidence=args.min_confidence,
            use_sliding_window=not args.no_sliding_window
        )
        
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 전체 처리 모드
        # 모델 정보에서 임계값 가져오기 (있는 경우)
        threshold_to_use = args.threshold
        if args.model_info:
            try:
                with open(args.model_info, 'r', encoding='utf-8') as f:
                    model_info_json = json.load(f)
                    optimal_threshold = model_info_json.get('optimal_threshold')
                    if optimal_threshold is not None:
                        threshold_to_use = optimal_threshold
                        print(f"모델 정보에서 최적 임계값 사용: {threshold_to_use:.4f}")
            except:
                pass
        
        process_all_files(
            input_dir=args.input_dir,
            model_path=args.model,
            model_info_path=args.model_info,
            output_dir=args.output_dir,
            window_height=args.window_height,
            window_degree=args.window_degree,
            stride_height=args.stride_height,
            stride_degree=args.stride_degree,
            threshold=threshold_to_use,
            min_confidence=args.min_confidence,
            use_sliding_window=not args.no_sliding_window,
            test_mode=test_mode,
            include_normal=args.include_normal,
            normal_sample_ratio=args.normal_sample_ratio
        )


if __name__ == "__main__":
    main()
