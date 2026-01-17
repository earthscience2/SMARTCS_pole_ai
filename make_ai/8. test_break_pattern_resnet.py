#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 ResNet 모델을 사용하여 테스트 데이터로 예측 및 평가를 수행하는 스크립트.

입력:
  - 모델: 6. models/break_pattern_resnet_best.keras
  - 모델 정보: 6. models/break_pattern_resnet_info.json
  - 테스트 데이터: 5. train_data/test/break_sequences_test.npy, break_labels_test.npy, break_positions_test.npy
출력:
  - 테스트 결과: 7. model_test/test_results.json
  - 시각화: 7. model_test/test_results.png
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
import platform

# 한글 폰트 설정 (OS별)
if platform.system() == 'Windows':
    try:
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    except:
        try:
            matplotlib.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕
        except:
            pass
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    matplotlib.rcParams['font.family'] = 'NanumGothic'

matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def find_latest_model_folder(models_dir: str = "6. models") -> Tuple[Path, str]:
    """
    가장 최근 날짜 폴더를 찾아서 반환.
    
    Args:
        models_dir: 모델 디렉토리 경로
    
    Returns:
        tuple: (최근 날짜 폴더 경로, 날짜 문자열)
    """
    models_path = Path(current_dir) / models_dir
    if not models_path.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {models_path}")
    
    # 날짜 폴더 찾기 (YYYYMMDD 또는 YYYYMMDD_HHMM 형식)
    date_folders = []
    for folder in models_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            # YYYYMMDD 형식 (8자리 숫자) 또는 YYYYMMDD_HHMM 형식 (13자리: 8자리_4자리)
            if len(folder_name) == 8 and folder_name.isdigit():
                try:
                    # 날짜 형식 검증
                    datetime.strptime(folder_name, "%Y%m%d")
                    date_folders.append((folder, folder_name))
                except ValueError:
                    continue
            elif len(folder_name) == 13 and '_' in folder_name:
                # YYYYMMDD_HHMM 형식 (예: 20260116_1430)
                try:
                    datetime.strptime(folder_name, "%Y%m%d_%H%M")
                    date_folders.append((folder, folder_name))
                except ValueError:
                    continue
    
    if not date_folders:
        raise FileNotFoundError(f"날짜 폴더를 찾을 수 없습니다: {models_path}")
    
    # 날짜 기준으로 정렬 (최신순)
    # 날짜 문자열을 datetime 객체로 변환하여 정확하게 정렬
    def get_datetime_from_folder_name(folder_name):
        if len(folder_name) == 8:
            return datetime.strptime(folder_name, "%Y%m%d")
        else:
            return datetime.strptime(folder_name, "%Y%m%d_%H%M")
    
    date_folders.sort(key=lambda x: get_datetime_from_folder_name(x[1]), reverse=True)
    latest_folder, latest_date = date_folders[0]
    
    print(f"가장 최근 모델 폴더: {latest_date} ({latest_folder})")
    return latest_folder, latest_date


def load_model_and_info(
    model_path: str,
    info_path: str
) -> Tuple[tf.keras.Model, dict]:
    """
    학습된 모델과 모델 정보 로드.
    
    Args:
        model_path: 모델 파일 경로 (.keras)
        info_path: 모델 정보 파일 경로 (.json)
    
    Returns:
        tuple: (모델, 모델 정보)
    """
    # 경로 변환
    if not os.path.isabs(model_path):
        model_path = os.path.join(current_dir, model_path)
    if not os.path.isabs(info_path):
        info_path = os.path.join(current_dir, info_path)
    
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = load_model(model_path, compile=False)
    
    # 모델 정보 로드
    print(f"모델 정보 로드 중: {info_path}")
    with open(info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    print(f"  모델 타입: {model_info.get('model_type', 'N/A')}")
    print(f"  시퀀스 길이: {model_info.get('sequence_length', 'N/A')}")
    print(f"  피처 수: {model_info.get('n_features', 'N/A')}")
    print(f"  최적 임계값: {model_info.get('optimal_threshold', 'N/A')}")
    
    return model, model_info


def load_test_data(
    sequences_path: str,
    labels_path: str,
    positions_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """
    테스트 데이터 로드.
    
    Args:
        sequences_path: 시퀀스 데이터 파일 경로
        labels_path: 라벨 데이터 파일 경로
        positions_path: 파단 위치 데이터 파일 경로 (선택사항)
        metadata_path: 메타데이터 파일 경로 (선택사항)
    
    Returns:
        tuple: (시퀀스 데이터, 라벨, 파단 위치, 메타데이터)
    """
    # 경로 변환
    if not os.path.isabs(sequences_path):
        sequences_path = os.path.join(current_dir, sequences_path)
    if not os.path.isabs(labels_path):
        labels_path = os.path.join(current_dir, labels_path)
    if positions_path and not os.path.isabs(positions_path):
        positions_path = os.path.join(current_dir, positions_path)
    if metadata_path and not os.path.isabs(metadata_path):
        metadata_path = os.path.join(current_dir, metadata_path)
    
    # 데이터 로드
    print(f"\n테스트 데이터 로드 중...")
    X_test = np.load(sequences_path)
    y_test = np.load(labels_path)
    
    positions_test = None
    if positions_path and os.path.exists(positions_path):
        positions_test = np.load(positions_path)
        print(f"  파단 위치 형태: {positions_test.shape}")
    
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print(f"  원본 데이터 형태: {X_test.shape}")
    
    # 2D 그리드 데이터인지 확인하고 1D로 변환
    is_2d_grid = metadata.get('data_type') == '2d_grid' or len(X_test.shape) == 4
    if is_2d_grid:
        print(f"  2D 그리드 데이터 감지: {X_test.shape}")
        # 2D 그리드를 1D로 변환 (모델 호환성을 위해)
        # (batch, height, degree, features) -> (batch, height*degree, features)
        if len(X_test.shape) == 4:
            batch_size, height_bins, degree_bins, n_features = X_test.shape
            X_test = X_test.reshape(batch_size, height_bins * degree_bins, n_features)
            print(f"  1D로 변환 완료: {X_test.shape} (height*degree를 시퀀스 길이로 사용)")
    
    print(f"  최종 데이터 형태: {X_test.shape}")
    print(f"  라벨 형태: {y_test.shape}")
    print(f"  클래스 분포: {np.bincount(y_test.astype(int))}")
    
    return X_test, y_test, positions_test, metadata


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict:
    """
    분류 성능 평가.
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률
    
    Returns:
        평가 지표 딕셔너리
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    
    # 분류 리포트
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc) if auc else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def evaluate_regression(
    y_true_positions: np.ndarray,
    y_pred_positions: np.ndarray,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray
) -> Dict:
    """
    회귀 성능 평가 (파단 위치 예측 오차).
    
    Args:
        y_true_positions: 실제 파단 위치 [height, degree]
        y_pred_positions: 예측 파단 위치 [height, degree]
        y_true_labels: 실제 라벨
        y_pred_labels: 예측 라벨
    
    Returns:
        평가 지표 딕셔너리
    """
    # 파단으로 예측된 샘플 또는 실제 파단 샘플에 대해서만 평가
    break_mask = (y_true_labels == 1) | (y_pred_labels == 1)
    
    if not np.any(break_mask):
        return {
            'mae': None,
            'mse': None,
            'rmse': None,
            'mae_height': None,
            'mae_degree': None,
            'mse_height': None,
            'mse_degree': None,
            'num_samples': 0
        }
    
    break_indices = np.where(break_mask)[0]
    true_positions = y_true_positions[break_indices]
    pred_positions = y_pred_positions[break_indices]
    
    # 정상 데이터는 [0.0, 0.0]이므로 제외
    valid_mask = np.any(true_positions > 0, axis=1)
    
    if not np.any(valid_mask):
        return {
            'mae': None,
            'mse': None,
            'rmse': None,
            'mae_height': None,
            'mae_degree': None,
            'mse_height': None,
            'mse_degree': None,
            'num_samples': 0
        }
    
    true_positions_valid = true_positions[valid_mask]
    pred_positions_valid = pred_positions[valid_mask]
    
    # 전체 오차
    mae = mean_absolute_error(true_positions_valid, pred_positions_valid)
    mse = mean_squared_error(true_positions_valid, pred_positions_valid)
    rmse = np.sqrt(mse)
    
    # 각 피처별 오차 (height, degree)
    mae_height = mean_absolute_error(true_positions_valid[:, 0], pred_positions_valid[:, 0])
    mae_degree = mean_absolute_error(true_positions_valid[:, 1], pred_positions_valid[:, 1])
    mse_height = mean_squared_error(true_positions_valid[:, 0], pred_positions_valid[:, 0])
    mse_degree = mean_squared_error(true_positions_valid[:, 1], pred_positions_valid[:, 1])
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae_height': float(mae_height),
        'mae_degree': float(mae_degree),
        'mse_height': float(mse_height),
        'mse_degree': float(mse_degree),
        'num_samples': int(np.sum(valid_mask))
    }


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    y_true_positions: Optional[np.ndarray],
    y_pred_positions: Optional[np.ndarray],
    threshold: float,
    output_path: Path
):
    """
    테스트 결과 시각화.
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률
        y_true_positions: 실제 파단 위치
        y_pred_positions: 예측 파단 위치
        threshold: 분류 임계값
        output_path: 출력 디렉토리
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 혼동 행렬
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('예측')
    ax1.set_ylabel('실제')
    ax1.set_title('혼동 행렬')
    
    # 2. ROC 곡선
    ax2 = plt.subplot(2, 3, 2)
    from sklearn.metrics import roc_curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC 곡선')
        ax2.legend()
        ax2.grid(True)
    
    # 3. 예측 확률 분포
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, label='정상', color='blue')
    ax3.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, label='파단', color='red')
    ax3.axvline(threshold, color='green', linestyle='--', label=f'임계값 ({threshold:.3f})')
    ax3.set_xlabel('예측 확률')
    ax3.set_ylabel('빈도')
    ax3.set_title('예측 확률 분포')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 파단 위치 예측 오차 (Height)
    if y_true_positions is not None and y_pred_positions is not None:
        break_mask = (y_true == 1) | (y_pred == 1)
        if np.any(break_mask):
            break_indices = np.where(break_mask)[0]
            true_pos = y_true_positions[break_indices]
            pred_pos = y_pred_positions[break_indices]
            valid_mask = np.any(true_pos > 0, axis=1)
            
            if np.any(valid_mask):
                true_pos_valid = true_pos[valid_mask]
                pred_pos_valid = pred_pos[valid_mask]
                
                # Height 오차
                ax4 = plt.subplot(2, 3, 4)
                height_error = np.abs(true_pos_valid[:, 0] - pred_pos_valid[:, 0])
                ax4.scatter(true_pos_valid[:, 0], pred_pos_valid[:, 0], alpha=0.5)
                ax4.plot([0, 1], [0, 1], 'r--', label='완벽한 예측')
                ax4.set_xlabel('실제 Height (정규화)')
                ax4.set_ylabel('예측 Height (정규화)')
                ax4.set_title(f'Height 예측 (MAE: {np.mean(height_error):.4f})')
                ax4.legend()
                ax4.grid(True)
                
                # Degree 오차
                ax5 = plt.subplot(2, 3, 5)
                degree_error = np.abs(true_pos_valid[:, 1] - pred_pos_valid[:, 1])
                ax5.scatter(true_pos_valid[:, 1], pred_pos_valid[:, 1], alpha=0.5)
                ax5.plot([0, 1], [0, 1], 'r--', label='완벽한 예측')
                ax5.set_xlabel('실제 Degree (정규화)')
                ax5.set_ylabel('예측 Degree (정규화)')
                ax5.set_title(f'Degree 예측 (MAE: {np.mean(degree_error):.4f})')
                ax5.legend()
                ax5.grid(True)
                
                # 오차 분포
                ax6 = plt.subplot(2, 3, 6)
                ax6.hist(height_error, bins=30, alpha=0.5, label='Height 오차', color='blue')
                ax6.hist(degree_error, bins=30, alpha=0.5, label='Degree 오차', color='red')
                ax6.set_xlabel('오차')
                ax6.set_ylabel('빈도')
                ax6.set_title('파단 위치 예측 오차 분포')
                ax6.legend()
                ax6.grid(True)
    
    plt.tight_layout()
    plot_file = output_path / "test_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 저장: {plot_file}")


def main():
    print("=" * 60)
    print("ResNet 모델 테스트 시작")
    print("=" * 60)
    
    # 가장 최근 날짜 폴더 찾기
    latest_folder, latest_date = find_latest_model_folder("6. models")
    
    # 모델 및 정보 파일 경로
    model_path = latest_folder / "break_pattern_resnet_best.keras"
    info_path = latest_folder / "break_pattern_resnet_info.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"모델 정보 파일을 찾을 수 없습니다: {info_path}")
    
    # 모델 및 정보 로드
    model, model_info = load_model_and_info(
        str(model_path),
        str(info_path)
    )
    
    # 테스트 데이터 로드
    X_test, y_test, positions_test, metadata = load_test_data(
        "5. train_data/test/break_sequences_test.npy",
        "5. train_data/test/break_labels_test.npy",
        "5. train_data/test/break_positions_test.npy",
        "5. train_data/break_sequences_metadata.json"
    )
    
    # 예측 수행
    print(f"\n예측 수행 중...")
    predictions = model.predict(X_test, verbose=1, batch_size=32)
    
    # 다중 출력 모델: [분류 출력, 회귀 출력]
    if isinstance(predictions, list):
        y_pred_proba = predictions[0].ravel()  # 분류 확률
        y_pred_positions = predictions[1]  # 파단 위치 예측 [height, degree]
    else:
        y_pred_proba = predictions.ravel()
        y_pred_positions = None
    
    # 임계값 설정
    threshold = model_info.get('optimal_threshold', 0.5)
    print(f"사용 임계값: {threshold:.4f}")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 분류 성능 평가
    print(f"\n분류 성능 평가:")
    print("-" * 60)
    classification_results = evaluate_classification(y_test, y_pred, y_pred_proba)
    
    print(f"  정확도: {classification_results['accuracy']:.4f}")
    print(f"  정밀도: {classification_results['precision']:.4f}")
    print(f"  재현율: {classification_results['recall']:.4f}")
    print(f"  F1 스코어: {classification_results['f1_score']:.4f}")
    if classification_results['auc']:
        print(f"  AUC: {classification_results['auc']:.4f}")
    
    print(f"\n혼동 행렬:")
    cm = np.array(classification_results['confusion_matrix'])
    print(f"  실제 정상 / 예측 정상: {cm[0, 0]}")
    print(f"  실제 정상 / 예측 파단: {cm[0, 1]}")
    print(f"  실제 파단 / 예측 정상: {cm[1, 0]}")
    print(f"  실제 파단 / 예측 파단: {cm[1, 1]}")
    
    # 회귀 성능 평가
    regression_results = {}
    if positions_test is not None and y_pred_positions is not None:
        print(f"\n회귀 성능 평가 (파단 위치 예측):")
        print("-" * 60)
        
        # 예측 값 통계 출력 (전체 데이터)
        print(f"\n예측 값 통계 (전체 테스트 데이터):")
        print(f"  Height 예측 범위: [{np.min(y_pred_positions[:, 0]):.4f}, {np.max(y_pred_positions[:, 0]):.4f}]")
        print(f"  Height 예측 평균: {np.mean(y_pred_positions[:, 0]):.4f}, 표준편차: {np.std(y_pred_positions[:, 0]):.4f}")
        print(f"  Degree 예측 범위: [{np.min(y_pred_positions[:, 1]):.4f}, {np.max(y_pred_positions[:, 1]):.4f}]")
        print(f"  Degree 예측 평균: {np.mean(y_pred_positions[:, 1]):.4f}, 표준편차: {np.std(y_pred_positions[:, 1]):.4f}")
        
        # 실제 파단 샘플에 대한 예측 값 통계
        break_indices = np.where(y_test == 1)[0]
        if len(break_indices) > 0:
            break_pred_positions = y_pred_positions[break_indices]
            print(f"\n예측 값 통계 (실제 파단 샘플 {len(break_indices)}개):")
            print(f"  Height 예측 범위: [{np.min(break_pred_positions[:, 0]):.4f}, {np.max(break_pred_positions[:, 0]):.4f}]")
            print(f"  Height 예측 평균: {np.mean(break_pred_positions[:, 0]):.4f}, 표준편차: {np.std(break_pred_positions[:, 0]):.4f}")
            print(f"  Degree 예측 범위: [{np.min(break_pred_positions[:, 1]):.4f}, {np.max(break_pred_positions[:, 1]):.4f}]")
            print(f"  Degree 예측 평균: {np.mean(break_pred_positions[:, 1]):.4f}, 표준편차: {np.std(break_pred_positions[:, 1]):.4f}")
            
            # 실제 값과 비교
            break_true_positions = positions_test[break_indices]
            valid_mask = np.any(break_true_positions > 0, axis=1)
            if np.any(valid_mask):
                print(f"\n실제 값 통계 (유효한 파단 샘플 {np.sum(valid_mask)}개):")
                valid_true = break_true_positions[valid_mask]
                valid_pred = break_pred_positions[valid_mask]
                print(f"  실제 Height 범위: [{np.min(valid_true[:, 0]):.4f}, {np.max(valid_true[:, 0]):.4f}]")
                print(f"  실제 Height 평균: {np.mean(valid_true[:, 0]):.4f}")
                print(f"  실제 Degree 범위: [{np.min(valid_true[:, 1]):.4f}, {np.max(valid_true[:, 1]):.4f}]")
                print(f"  실제 Degree 평균: {np.mean(valid_true[:, 1]):.4f}")
                print(f"\n  예측 Height 범위: [{np.min(valid_pred[:, 0]):.4f}, {np.max(valid_pred[:, 0]):.4f}]")
                print(f"  예측 Height 평균: {np.mean(valid_pred[:, 0]):.4f}")
                print(f"  예측 Degree 범위: [{np.min(valid_pred[:, 1]):.4f}, {np.max(valid_pred[:, 1]):.4f}]")
                print(f"  예측 Degree 평균: {np.mean(valid_pred[:, 1]):.4f}")
        
        regression_results = evaluate_regression(
            positions_test, y_pred_positions, y_test, y_pred
        )
        
        if regression_results['num_samples'] > 0:
            print(f"\n회귀 성능 지표:")
            print(f"  평가 샘플 수: {regression_results['num_samples']}개")
            print(f"  전체 MAE: {regression_results['mae']:.4f}")
            print(f"  전체 MSE: {regression_results['mse']:.4f}")
            print(f"  전체 RMSE: {regression_results['rmse']:.4f}")
            print(f"  Height MAE: {regression_results['mae_height']:.4f}")
            print(f"  Degree MAE: {regression_results['mae_degree']:.4f}")
            print(f"  Height MSE: {regression_results['mse_height']:.4f}")
            print(f"  Degree MSE: {regression_results['mse_degree']:.4f}")
        else:
            print("  평가할 파단 샘플이 없습니다.")
    
    # 결과 저장 (같은 날짜 폴더에 저장)
    output_path = latest_folder  # 모델이 있는 날짜 폴더에 결과도 저장
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n결과 저장 경로: {output_path} (날짜: {latest_date})")
    
    test_results = {
        'model_info': {
            'model_type': model_info.get('model_type'),
            'sequence_length': model_info.get('sequence_length'),
            'n_features': model_info.get('n_features'),
            'optimal_threshold': float(threshold),
        },
        'test_data': {
            'num_samples': int(len(X_test)),
            'class_distribution': {
                'normal': int(np.sum(y_test == 0)),
                'break': int(np.sum(y_test == 1))
            }
        },
        'classification': classification_results,
        'regression': regression_results,
        'predictions': {
            'y_pred_proba_mean': float(np.mean(y_pred_proba)),
            'y_pred_proba_std': float(np.std(y_pred_proba)),
        },
        'regression_predictions': {
            'height_min': float(np.min(y_pred_positions[:, 0])) if y_pred_positions is not None else None,
            'height_max': float(np.max(y_pred_positions[:, 0])) if y_pred_positions is not None else None,
            'height_mean': float(np.mean(y_pred_positions[:, 0])) if y_pred_positions is not None else None,
            'height_std': float(np.std(y_pred_positions[:, 0])) if y_pred_positions is not None else None,
            'degree_min': float(np.min(y_pred_positions[:, 1])) if y_pred_positions is not None else None,
            'degree_max': float(np.max(y_pred_positions[:, 1])) if y_pred_positions is not None else None,
            'degree_mean': float(np.mean(y_pred_positions[:, 1])) if y_pred_positions is not None else None,
            'degree_std': float(np.std(y_pred_positions[:, 1])) if y_pred_positions is not None else None,
        } if y_pred_positions is not None else None
    }
    
    results_file = output_path / "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"결과 저장: {results_file}")
    
    # 시각화
    print(f"\n시각화 생성 중...")
    plot_results(
        y_test, y_pred, y_pred_proba,
        positions_test, y_pred_positions,
        threshold, output_path
    )
    
    print("\n" + "=" * 60)
    print("ResNet 모델 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
