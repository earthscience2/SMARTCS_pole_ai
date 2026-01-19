#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시퀀스 데이터를 이용하여 파단 패턴을 학습하는 ResNet 모델 학습 스크립트.

입력: 6. train_data/sequences/break_sequences.npy, break_labels.npy
출력: 6. models/break_pattern_resnet_best.keras (학습된 모델)

4. edit_pole_data의 파단 위치와 파단 데이터, 정상 데이터를 이용하여 학습합니다.
"""

import os
import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add,
    MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,
    AveragePooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
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


def load_sequence_data(
    sequences_path: str,
    labels_path: str,
    metadata_path: Optional[str] = None,
    positions_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """
    시퀀스 데이터 로드.
    
    Args:
        sequences_path: 시퀀스 데이터 파일 경로 (.npy)
        labels_path: 라벨 데이터 파일 경로 (.npy)
        metadata_path: 메타데이터 파일 경로 (.json, 선택사항)
        positions_path: 파단 위치 데이터 파일 경로 (.npy, 선택사항)
    
    Returns:
        tuple: (시퀀스 데이터, 라벨, 파단 위치, 메타데이터)
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(sequences_path):
        sequences_path = os.path.join(current_dir, sequences_path)
    if not os.path.isabs(labels_path):
        labels_path = os.path.join(current_dir, labels_path)
    if metadata_path and not os.path.isabs(metadata_path):
        metadata_path = os.path.join(current_dir, metadata_path)
    if positions_path and not os.path.isabs(positions_path):
        positions_path = os.path.join(current_dir, positions_path)
    
    X = np.load(sequences_path)
    y = np.load(labels_path)
    
    # 메타데이터 먼저 로드
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # 파단 위치 또는 윈도우 위치 데이터 로드
    positions = None
    if positions_path and os.path.exists(positions_path):
        positions = np.load(positions_path)
        print(f"  위치 데이터 형태: {positions.shape}")
        # window_positions인 경우 메타데이터에서 확인
        if metadata.get('data_type') == 'sliding_window':
            print(f"  윈도우 위치 데이터 (center_height, center_degree)")
        else:
            print(f"  파단 위치 데이터 (break_height, break_degree)")
    
    print(f"데이터 로드 완료:")
    print(f"  데이터 형태: {X.shape}")
    print(f"  라벨 형태: {y.shape}")
    print(f"  클래스 분포: {np.bincount(y.astype(int))}")
    
    # 데이터 타입 확인
    data_type = metadata.get('data_type', 'unknown')
    
    # 2D 그리드 데이터인지 확인
    is_2d_grid = data_type == '2d_grid' or len(X.shape) == 4
    if is_2d_grid:
        print(f"  데이터 타입: 2D 그리드 (height × degree × features)")
        if 'grid_shape' in metadata:
            print(f"  그리드 형태: {metadata['grid_shape']}")
        # 2D 그리드를 1D로 변환 (모델 호환성을 위해)
        # (batch, height, degree, features) -> (batch, height*degree, features)
        if len(X.shape) == 4:
            batch_size, height_bins, degree_bins, n_features = X.shape
            X = X.reshape(batch_size, height_bins * degree_bins, n_features)
            print(f"  1D로 변환: {X.shape} (height*degree를 시퀀스 길이로 사용)")
    elif data_type == 'sliding_window':
        # 슬라이딩 윈도우 방식
        print(f"  데이터 타입: 슬라이딩 윈도우 (1D 시퀀스)")
        print(f"    시퀀스 길이: {metadata.get('sequence_length', X.shape[1] if len(X.shape) >= 2 else 'N/A')}")
        print(f"    윈도우 크기: 높이 {metadata.get('window_height', 'N/A')}m, 각도 {metadata.get('window_degree', 'N/A')}°")
        print(f"    스트라이드: 높이 {metadata.get('stride_height', 'N/A')}m, 각도 {metadata.get('stride_degree', 'N/A')}°")
        print(f"    크롭 마진: 높이 ±{metadata.get('crop_height_margin', 'N/A')}m, 각도 ±{metadata.get('crop_degree_margin', 'N/A')}°")
        print(f"    라벨 의미: 1=파단 위치 포함 윈도우, 0=파단 위치 미포함 윈도우")
    else:
        print(f"  데이터 타입: 1D 시퀀스")
        # 슬라이딩 윈도우 사용 여부 확인
        if metadata.get('use_sliding_window', False):
            print(f"  데이터 준비 방식: 슬라이딩 윈도우 (평가 시와 동일)")
            print(f"    윈도우 크기: 높이 {metadata.get('window_height', 'N/A')}m, 각도 {metadata.get('window_degree', 'N/A')}°")
            print(f"    스트라이드: 높이 {metadata.get('stride_height', 'N/A')}m, 각도 {metadata.get('stride_degree', 'N/A')}°")
        else:
            print(f"  데이터 준비 방식: 크롭된 영역 직접 사용")
    
    return X, y, positions, metadata


def residual_block_1d(x, filters, kernel_size=3, stride=1, regularizer=None):
    """
    1D Residual Block 구현.
    
    Args:
        x: 입력 텐서
        filters: 필터 수
        kernel_size: 커널 크기
        stride: 스트라이드
        regularizer: 정규화 계수
    
    Returns:
        출력 텐서
    """
    shortcut = x
    
    # Main path
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', 
               kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size, padding='same', 
               kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same',
                          kernel_regularizer=regularizer)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_resnet_1d(
    sequence_length: int,
    n_features: int,
    num_filters: int = 64,
    num_blocks: int = 3,
    blocks_per_layer: int = 2,
    kernel_size: int = 7,
    dropout_rate: float = 0.3,
    regularizer: Optional[float] = None,
    use_pooling: bool = True
) -> Model:
    """
    1D ResNet 모델 구성.
    
    Args:
        sequence_length: 시퀀스 길이
        n_features: 피처 수
        num_filters: 초기 필터 수
        num_blocks: Residual block 레이어 수
        blocks_per_layer: 각 레이어당 블록 수
        kernel_size: 초기 컨볼루션 커널 크기
        dropout_rate: Dropout 비율
        regularizer: 정규화 계수
        use_pooling: MaxPooling 사용 여부
    
    Returns:
        Keras 모델
    """
    inputs = Input(shape=(sequence_length, n_features))
    
    reg = l1_l2(l1=regularizer, l2=regularizer) if regularizer else None
    
    # Initial convolution
    x = Conv1D(num_filters, kernel_size=kernel_size, padding='same',
               kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if use_pooling:
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    filters = num_filters
    for i in range(num_blocks):
        # 각 레이어의 첫 번째 블록에서 stride=2로 다운샘플링
        stride = 2 if i > 0 else 1
        
        for j in range(blocks_per_layer):
            if j == 0 and stride > 1:
                x = residual_block_1d(x, filters, stride=stride, regularizer=reg)
            else:
                x = residual_block_1d(x, filters, stride=1, regularizer=reg)
        
        # 다음 레이어로 필터 수 증가
        if i < num_blocks - 1:
            filters *= 2
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Shared feature extraction (분류 및 회귀 모두에 사용되는 공유 특징)
    # 정확도 향상을 위해 더 깊고 넓은 네트워크 사용
    # 회귀 예측의 다양성을 위해 충분한 특징 추출
    shared = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout_rate * 0.8)(shared)  # Dropout 약간 감소 (과도한 정규화 방지)
    
    shared = Dense(128, activation='relu', kernel_regularizer=reg)(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout_rate * 0.6)(shared)
    
    shared = Dense(64, activation='relu', kernel_regularizer=reg)(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout_rate * 0.4)(shared)
    
    # Classification head (정상/파단 분류)
    # False Negative 감소를 위해 더 깊고 넓은 네트워크 사용
    classification = Dense(128, activation='relu', kernel_regularizer=reg)(shared)
    classification = BatchNormalization()(classification)
    classification = Dropout(dropout_rate * 0.4)(classification)  # Dropout 감소 (과도한 정규화 방지)
    
    classification = Dense(96, activation='relu', kernel_regularizer=reg)(classification)
    classification = BatchNormalization()(classification)
    classification = Dropout(dropout_rate * 0.3)(classification)
    
    classification = Dense(64, activation='relu', kernel_regularizer=reg)(classification)
    classification = BatchNormalization()(classification)
    classification = Dropout(dropout_rate * 0.2)(classification)
    
    classification = Dense(32, activation='relu', kernel_regularizer=reg)(classification)
    classification = BatchNormalization()(classification)
    classification_output = Dense(1, activation='sigmoid', name='classification')(classification)
    
    # Regression head (파단 위치 예측: height, degree)
    # 파단 위치 예측 정밀도 향상을 위해 더 깊고 넓은 네트워크 사용
    # Dropout을 더 줄여서 위치 예측 다양성 향상
    regression = Dense(128, activation='relu', kernel_regularizer=reg)(shared)
    regression = BatchNormalization()(regression)
    regression = Dropout(dropout_rate * 0.3)(regression)  # Dropout 감소 (0.4 -> 0.3, 위치 예측 다양성 향상)
    
    regression = Dense(96, activation='relu', kernel_regularizer=reg)(regression)
    regression = BatchNormalization()(regression)
    regression = Dropout(dropout_rate * 0.2)(regression)  # Dropout 감소 (0.3 -> 0.2)
    
    regression = Dense(64, activation='relu', kernel_regularizer=reg)(regression)
    regression = BatchNormalization()(regression)
    regression = Dropout(dropout_rate * 0.1)(regression)  # Dropout 감소 (0.2 -> 0.1)
    
    regression = Dense(32, activation='relu', kernel_regularizer=reg)(regression)
    regression = BatchNormalization()(regression)
    # 마지막 레이어는 Dropout 없음 (정밀도 향상)
    
    # 최종 출력 레이어 - 위치 예측의 정밀도를 위해 더 넓은 레이어 사용
    regression_output = Dense(2, activation='linear', name='regression')(regression)  # 정규화된 값 (0~1 범위지만 linear로 학습)
    
    model = Model(inputs=inputs, outputs=[classification_output, regression_output])
    return model


def calculate_class_weights(y: np.ndarray) -> dict:
    """
    클래스 가중치 계산 (불균형 데이터 처리).
    
    Args:
        y: 라벨 배열
    
    Returns:
        클래스 가중치 딕셔너리
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return {i: weight for i, weight in zip(classes, class_weights)}


def weighted_binary_crossentropy(class_weight: dict):
    """
    클래스 가중치가 적용된 binary crossentropy loss 함수.
    
    Args:
        class_weight: 클래스 가중치 딕셔너리
    
    Returns:
        가중치가 적용된 loss 함수
    """
    # 클래스 가중치를 float32로 변환
    weight_0 = tf.constant(float(class_weight[0]), dtype=tf.float32)
    weight_1 = tf.constant(float(class_weight[1]), dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        # 기본 binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # 클래스 가중치 적용 (float32로 변환)
        weights = tf.where(y_true == 1, weight_1, weight_0)
        weighted_bce = bce * weights
        
        return tf.reduce_mean(weighted_bce)
    
    return loss_fn


def focal_loss_with_false_negative_penalty(class_weight: dict, alpha: float = 0.25, gamma: float = 2.0, fn_penalty: float = 2.0):
    """
    Focal Loss + False Negative 페널티가 적용된 loss 함수.
    False Negative (파단을 정상으로 오분류)에 더 큰 페널티를 부여하여 Recall을 높임.
    
    목표: Recall >= 95% (FN <= 5%), Precision >= 50%
    
    Args:
        class_weight: 클래스 가중치 딕셔너리
        alpha: 클래스 균형 가중치 (0~1)
        gamma: Focal Loss focusing 파라미터 (클수록 어려운 샘플에 집중)
        fn_penalty: False Negative 추가 페널티 배수 (클수록 FN에 더 큰 페널티)
    
    Returns:
        Focal Loss + False Negative 페널티가 적용된 loss 함수
    """
    weight_0 = tf.constant(float(class_weight[0]), dtype=tf.float32)
    weight_1 = tf.constant(float(class_weight[1]), dtype=tf.float32)
    alpha_t = tf.constant(alpha, dtype=tf.float32)
    gamma_t = tf.constant(gamma, dtype=tf.float32)
    fn_penalty_t = tf.constant(fn_penalty, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        # 기본 binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Focal Loss 계산
        # p_t: 예측 확률 (정확한 클래스에 대한)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0 - 1e-7)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, gamma_t)
        
        # Alpha weight: 클래스 가중치
        alpha_weight = y_true * weight_1 * alpha_t + (1 - y_true) * weight_0 * (1 - alpha_t)
        
        # Focal Loss
        focal_loss = bce * focal_weight * alpha_weight
        
        # False Negative 추가 페널티
        # y_true=1 (파단)이고 y_pred가 낮으면 (정상으로 예측) 큰 페널티
        fn_mask = tf.cast(y_true == 1, tf.float32)  # 실제 파단인 경우
        fn_error = (1.0 - y_pred) * fn_mask  # 파단인데 정상으로 예측한 정도
        fn_penalty_term = fn_error * fn_penalty_t
        
        # 최종 loss: Focal Loss + False Negative 페널티
        total_loss = focal_loss + fn_penalty_term * bce
        
        return tf.reduce_mean(total_loss)
    
    return loss_fn


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prioritize_recall: bool = True
) -> float:
    """
    최적 임계값 찾기 (재현율 우선 또는 F1 스코어 최대화).
    
    Args:
        y_true: 실제 라벨
        y_pred_proba: 예측 확률
        prioritize_recall: 재현율 우선 여부
    
    Returns:
        최적 임계값
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # precision_recall_curve의 반환값 특성:
    # - precision, recall 길이: N
    # - thresholds 길이: N-1
    # 마지막 포인트(precision[-1], recall[-1])는 threshold와 매칭되지 않으므로 제거 후 사용
    if len(thresholds) + 1 == len(precision):
        precision = precision[:-1]
        recall = recall[:-1]
    
    if prioritize_recall:
        # 목표: Recall >= 95% (FN <= 5%), Precision >= 50%
        # 재현율이 95% 이상이고 정밀도가 50% 이상인 임계값 중에서 F1 스코어가 가장 높은 것 선택
        valid_indices = (recall >= 0.95) & (precision >= 0.50)
        if np.any(valid_indices):
            # F1 스코어 계산
            f1_scores = 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices] + 1e-10)
            best_idx = np.argmax(f1_scores)
            return thresholds[valid_indices][best_idx]
        else:
            # 재현율 95% 이상인 임계값 중에서 정밀도를 고려하여 선택
            valid_indices = recall >= 0.95
            if np.any(valid_indices):
                # 재현율 95% 이상 중에서 정밀도가 가장 높은 것 선택
                best_idx = np.argmax(precision[valid_indices])
                return thresholds[valid_indices][best_idx]
            else:
                # 재현율 90% 이상인 임계값 중에서 F1 스코어 최대화
                valid_indices = recall >= 0.90
                if np.any(valid_indices):
                    f1_scores = 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices] + 1e-10)
                    best_idx = np.argmax(f1_scores)
                    return thresholds[valid_indices][best_idx]
                else:
                    # 재현율이 낮으면 F1 스코어 최대화
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    best_idx = np.argmax(f1_scores)
                    return thresholds[best_idx]
    else:
        # F1 스코어 최대화
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    positions_train: Optional[np.ndarray] = None,
    positions_val: Optional[np.ndarray] = None,
    num_filters: int = 64,
    num_blocks: int = 3,
    blocks_per_layer: int = 2,
    kernel_size: int = 7,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    regularizer: Optional[float] = None,
    output_dir: str = "6. models",
    use_class_weight: bool = True,
    optimal_threshold: bool = True,
    regression_weight: float = 0.5,
    metadata: Optional[dict] = None
) -> Tuple[Model, dict, float]:
    """
    ResNet 모델 학습 (분류 + 회귀).
    
    Args:
        X_train: 학습 시퀀스 데이터
        y_train: 학습 라벨
        X_val: 검증 시퀀스 데이터
        y_val: 검증 라벨
        positions_train: 학습 파단 위치 데이터 (정규화된 [height, degree])
        positions_val: 검증 파단 위치 데이터 (정규화된 [height, degree])
        num_filters: 초기 필터 수
        num_blocks: Residual block 레이어 수
        blocks_per_layer: 각 레이어당 블록 수
        kernel_size: 초기 컨볼루션 커널 크기
        dropout_rate: Dropout 비율
        learning_rate: 학습률
        batch_size: 배치 크기
        epochs: 에폭 수
        regularizer: 정규화 계수
        output_dir: 모델 저장 디렉토리
        use_class_weight: 클래스 가중치 사용 여부
        optimal_threshold: 최적 임계값 찾기 여부
        regression_weight: 회귀 loss 가중치 (기본값: 0.5)
        metadata: 메타데이터
    
    Returns:
        tuple: (학습된 모델, 학습 정보, 최적 임계값)
    """
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # 클래스 가중치 계산 (모델 컴파일 전에 필요)
    class_weight = None
    if use_class_weight:
        class_weight = calculate_class_weights(y_train)
        print(f"\n클래스 가중치: {class_weight}")
    
    # 모델 구성
    model = build_resnet_1d(
        sequence_length=sequence_length,
        n_features=n_features,
        num_filters=num_filters,
        num_blocks=num_blocks,
        blocks_per_layer=blocks_per_layer,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        regularizer=regularizer
    )
    
    # 모델 컴파일 (다중 출력: 분류 + 회귀)
    optimizer = Adam(learning_rate=learning_rate)
    
    # False Negative를 줄이기 위한 Focal Loss + False Negative 페널티 사용
    # 목표: Recall >= 95% (FN <= 5%), Precision >= 50%, Accuracy >= 90%
    classification_loss = 'binary_crossentropy'
    if class_weight and positions_train is not None:
        # 다중 출력 모델: Focal Loss + False Negative 페널티 적용
        # Recall 95% 목표를 위해 FN 페널티를 더 강화 (3.0 -> 5.0)
        # Precision 50% 유지를 위해 alpha를 적절히 조정 (0.30 -> 0.35)
        classification_loss = focal_loss_with_false_negative_penalty(
            class_weight, 
            alpha=0.35,  # 0.30 -> 0.35: 파단 클래스에 더 집중하여 Precision 향상
            gamma=2.5,   # 어려운 샘플에 집중 (유지)
            fn_penalty=5.0  # 3.0 -> 5.0: False Negative에 더 큰 페널티 (Recall 95% 목표)
        )
        print(f"  Focal Loss + False Negative 페널티 사용 (Recall >= 95%, Precision >= 50% 목표)")
        print(f"    - Focal Loss (gamma=2.5): 어려운 샘플에 집중")
        print(f"    - False Negative 페널티 (5.0배): 파단을 정상으로 오분류 시 큰 페널티 (Recall 95% 목표)")
        print(f"    - Alpha (0.35): 파단 클래스에 집중하여 Precision 50% 유지")
    
    # 파단 위치 데이터가 있는 경우 회귀 loss 포함
    if positions_train is not None:
        # Huber loss 기반 회귀 loss (각도 예측 개선을 위해 각도에 더 큰 가중치)
        def masked_huber_loss_with_degree_weight(y_true, y_pred, delta=0.2, degree_weight=1.5):
            """
            정상 데이터(파단 위치가 [0, 0])는 회귀 loss에서 제외
            파단 데이터만 Huber loss 계산 (각도 예측 개선을 위해 각도에 더 큰 가중치)
            
            Args:
                y_true: 실제 파단 위치 [height, degree]
                y_pred: 예측 파단 위치 [height, degree]
                delta: Huber loss 임계값 (기본값: 0.2)
                degree_weight: 각도 오차에 대한 가중치 (기본값: 1.5, 각도 예측 개선)
            """
            # 파단 위치가 유효한지 확인 (height나 degree가 0보다 큰 경우)
            valid_mask = tf.cast(tf.reduce_sum(y_true, axis=1) > 0, tf.float32)
            
            # 오차 계산
            error = y_true - y_pred  # [height_error, degree_error]
            abs_error = tf.abs(error)
            
            # Huber loss: 작은 오차는 MSE, 큰 오차는 MAE
            # L_delta(a) = 0.5 * a^2 if |a| <= delta, else delta * |a| - 0.5 * delta^2
            squared_loss = 0.5 * tf.square(error)
            linear_loss = delta * abs_error - 0.5 * tf.square(delta)
            huber_loss = tf.where(abs_error <= delta, squared_loss, linear_loss)
            
            # 각도 예측 개선을 위해 각도 오차에 더 큰 가중치 부여
            # huber_loss shape: [batch_size, 2] (height, degree)
            height_loss = huber_loss[:, 0]  # 높이 오차
            degree_loss = huber_loss[:, 1] * degree_weight  # 각도 오차에 가중치 적용
            
            # 가중치가 적용된 loss (각도에 더 집중)
            # 각도와 높이의 가중 평균 (각도에 더 큰 비중)
            total_weight = 1.0 + degree_weight
            weighted_huber_per_sample = (height_loss + degree_loss) / total_weight
            
            # 유효한 샘플만 선택
            masked_huber = weighted_huber_per_sample * valid_mask
            
            # 유효한 샘플이 있는 경우에만 평균 계산
            valid_count = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
            return tf.reduce_sum(masked_huber) / valid_count
        
        # 각도 예측 개선을 위한 가중치가 적용된 Huber loss 함수 생성
        # 각도 예측 정확도 향상을 위해 각도 가중치 증가
        masked_huber = lambda y_true, y_pred: masked_huber_loss_with_degree_weight(
            y_true, y_pred, delta=0.2, degree_weight=2.0  # 각도 가중치 증가 (1.5 -> 2.0)
        )
        
        model.compile(
            optimizer=optimizer,
            loss={
                'classification': classification_loss,
                'regression': masked_huber  # 각도 예측 개선을 위한 가중치가 적용된 Huber loss
            },
            loss_weights={
                'classification': 1.0,
                'regression': regression_weight * 7.0  # 회귀 loss 가중치 (7배, 1618, 1636 모델과 유사한 수준)
            },
            metrics={
                'classification': ['accuracy', 'precision', 'recall'],
                'regression': ['mae', 'mse']
            }
        )
        print(f"  회귀 loss: 가중치가 적용된 Huber loss (각도 가중치 2.0배)")
        print(f"    - Height와 Degree를 별도로 계산하여 각도 오차에 2.0배 가중치 적용")
        print(f"    - 각도 예측 정확도 향상에 집중")
        print(f"  회귀 loss 가중치: {regression_weight * 7.0} (기본값의 7배, 1618, 1636 모델과 유사한 수준)")
        print(f"  회귀 헤드: 128->96->64->32->2 (Dropout 감소로 위치 예측 다양성 및 정밀도 향상)")
    else:
        # 파단 위치 데이터가 없는 경우 분류만
        # 목표: Recall >= 95% (FN <= 5%), Precision >= 50%, Accuracy >= 90%
        if class_weight:
            # False Negative를 줄이기 위한 Focal Loss + False Negative 페널티 사용
            classification_loss = focal_loss_with_false_negative_penalty(
                class_weight,
                alpha=0.35,  # 0.30 -> 0.35: 파단 클래스에 더 집중하여 Precision 향상
                gamma=2.5,   # 어려운 샘플에 집중 (유지)
                fn_penalty=5.0  # 3.0 -> 5.0: False Negative에 더 큰 페널티 (Recall 95% 목표)
            )
            print(f"  Focal Loss + False Negative 페널티 사용 (Recall >= 95%, Precision >= 50% 목표)")
            print(f"    - Focal Loss (gamma=2.5): 어려운 샘플에 집중")
            print(f"    - False Negative 페널티 (5.0배): 파단을 정상으로 오분류 시 큰 페널티 (Recall 95% 목표)")
            print(f"    - Alpha (0.35): 파단 클래스에 집중하여 Precision 50% 유지")
        model.compile(
            optimizer=optimizer,
            loss={'classification': classification_loss},
            metrics={'classification': ['accuracy', 'precision', 'recall']}
        )
        print(f"  파단 위치 데이터 없음: 분류만 수행")
    
    print(f"\n모델 구성 완료:")
    model.summary()
    
    # 출력 디렉토리 생성 (날짜 폴더 포함)
    base_output_path = Path(current_dir) / output_dir
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    # 날짜 폴더 생성 (YYYYMMDD_HHMM 형식 - 년월일_시분)
    date_folder = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = base_output_path / date_folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n결과 저장 경로: {output_path}")
    
    # 콜백 설정
    # False Negative를 줄이기 위해 val_classification_recall을 모니터링
    model_file = output_path / "break_pattern_resnet_best.keras"
    
    # 목표: Recall >= 95% (FN <= 5%), Precision >= 50%, Accuracy >= 90%
    # Recall을 우선적으로 보장하면서 Precision도 50% 이상 유지
    # Recall을 모니터링하되, Precision도 함께 고려
    if positions_train is not None:
        # Recall 모니터링으로 False Negative 최소화 (Recall >= 95% 목표)
        # 하지만 Precision도 50% 이상 유지해야 하므로 Accuracy도 함께 고려
        monitor_metric = 'val_classification_recall'  # Recall 모니터링 (FN <= 5% 목표)
        recall_monitor = 'val_classification_recall'
    else:
        # 단일 출력 모델의 경우 Recall 모니터링
        monitor_metric = 'val_recall'  # Recall 모니터링 (FN <= 5% 목표)
        recall_monitor = 'val_recall'
    
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,  # Recall 모니터링 (FN <= 5% 목표)
            mode='max',  # Recall은 높을수록 좋음
            patience=25,  # 더 긴 학습 허용 (Recall 95% 목표 달성)
            restore_best_weights=True,
            verbose=0  # 1 -> 0: 출력 최소화로 속도 향상
        ),
        ModelCheckpoint(
            str(model_file),
            monitor=monitor_metric,  # Recall이 가장 높은 모델 저장 (FN <= 5% 목표)
            mode='max',
            save_best_only=True,
            verbose=0  # 1 -> 0: 출력 최소화로 속도 향상
        ),
        ReduceLROnPlateau(
            monitor='val_loss',  # Learning rate는 여전히 loss로 조정
            factor=0.3,  # 0.5 -> 0.3: 더 적극적인 학습률 감소
            patience=5,  # 7 -> 5: 더 빠른 학습률 조정
            min_lr=1e-7,  # 1e-8 -> 1e-7: 최소 학습률 상향
            verbose=0  # 1 -> 0: 출력 최소화로 속도 향상
        ),
    ]
    print(f"  EarlyStopping 및 ModelCheckpoint: {monitor_metric} 모니터링 (Recall >= 95% 목표)")
    print(f"  학습률 스케줄링: factor=0.3, patience=5, min_lr=1e-7 (더 적극적인 학습률 감소)")
    print(f"  목표 성능: Accuracy >= 90%, Recall >= 95% (FN <= 5%), Precision >= 50%")
    
    # 모델 학습
    print(f"\n모델 학습 시작...")
    print(f"  학습 샘플: {len(X_train)}개")
    print(f"  검증 샘플: {len(X_val)}개")
    print(f"  배치 크기: {batch_size}")
    print(f"  최대 에폭: {epochs}")
    if class_weight:
        print(f"  클래스 가중치 적용: {class_weight}")
    
    # 학습 시작 시간 기록
    train_start_time = time.time()
    train_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 학습 데이터 준비 (다중 출력)
    if positions_train is not None:
        y_train_dict = {
            'classification': y_train,
            'regression': positions_train
        }
        y_val_dict = {
            'classification': y_val,
            'regression': positions_val
        }
    else:
        y_train_dict = {'classification': y_train}
        y_val_dict = {'classification': y_val}
    
    # 다중 출력 모델에서는 커스텀 loss 함수에 클래스 가중치가 포함되어 있음
    # 단일 출력 모델에서만 class_weight 사용
    class_weight_dict = None
    if class_weight and positions_train is None:
        # 단일 출력 모델: class_weight 사용 가능
        class_weight_dict = {'classification': class_weight}
    
    # 학습 속도 향상을 위해 verbose 모드 조정
    # verbose=1: 진행 표시줄 (기본)
    # verbose=2: 에폭당 한 줄 (더 빠름)
    history = model.fit(
        X_train, y_train_dict,
        validation_data=(X_val, y_val_dict),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2  # 1 -> 2: 에폭당 한 줄만 출력하여 속도 향상
    )
    
    # 학습 종료 시간 기록
    train_end_time = time.time()
    train_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_duration = train_end_time - train_start_time
    train_duration_hours = train_duration / 3600
    
    # 최적 모델 로드
    model.load_weights(str(model_file))
    
    # 검증 성능 평가 (verbose=0으로 출력 최소화)
    predictions = model.predict(X_val, verbose=0, batch_size=batch_size * 2)  # 배치 크기 증가로 속도 향상
    
    # 다중 출력 모델: [분류 출력, 회귀 출력]
    if isinstance(predictions, list):
        y_pred_proba = predictions[0].ravel()  # 분류 확률
        y_pred_positions = predictions[1]  # 파단 위치 예측 [height, degree]
    else:
        y_pred_proba = predictions.ravel()
        y_pred_positions = None
    
    # 최적 임계값 찾기
    # 목표: Recall >= 95% (FN <= 5%), Precision >= 50%, Accuracy >= 90%
    threshold = 0.5
    if optimal_threshold:
        # Recall >= 95%, Precision >= 50%를 만족하는 임계값 선택
        threshold = find_optimal_threshold(y_val, y_pred_proba, prioritize_recall=True)
        print(f"\n최적 임계값: {threshold:.4f} (목표: Recall >= 95%, Precision >= 50%)")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 분류 성능 평가
    val_accuracy = accuracy_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred, zero_division=0)
    val_recall = recall_score(y_val, y_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_pred, zero_division=0)
    
    val_auc = None
    if len(np.unique(y_val)) > 1:
        try:
            val_auc = roc_auc_score(y_val, y_pred_proba)
        except:
            pass
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_val, y_pred)
    # 혼동 행렬이 2x2인 경우만 TN, FP, FN, TP 추출
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # 2x2가 아닌 경우 (클래스가 1개만 있는 경우 등)
        if cm.shape == (1, 1):
            if y_val[0] == 0:  # 정상만 있는 경우
                tn, fp, fn, tp = (cm[0, 0], 0, 0, 0)
            else:  # 파단만 있는 경우
                tn, fp, fn, tp = (0, 0, 0, cm[0, 0])
        else:
            tn, fp, fn, tp = (0, 0, 0, 0)
    
    # 클래스별 상세 성능 지표
    classification_report_dict = classification_report(
        y_val, y_pred, 
        target_names=['정상 (Normal)', '파단 (Break)'],
        output_dict=True,
        zero_division=0
    )
    
    # ROC 곡선 데이터
    roc_curve_data = None
    if len(np.unique(y_val)) > 1:
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_val, y_pred_proba)
            roc_curve_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
        except:
            pass
    
    # Precision-Recall 곡선 데이터
    pr_curve_data = None
    try:
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, y_pred_proba)
        pr_curve_data = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
    except:
        pass
    
    # 회귀 성능 평가 (파단 샘플에 대해서만)
    val_regression_mae = None
    val_regression_mse = None
    regression_error_details = None
    if positions_val is not None and y_pred_positions is not None:
        # 파단으로 예측된 샘플 또는 실제 파단 샘플에 대해서만 평가
        break_mask = (y_val == 1) | (y_pred == 1)
        if np.any(break_mask):
            break_indices = np.where(break_mask)[0]
            true_positions = positions_val[break_indices]
            pred_positions = y_pred_positions[break_indices]
            
            # 정상 데이터는 [0.0, 0.0]이므로 제외
            valid_mask = np.any(true_positions > 0, axis=1)
            if np.any(valid_mask):
                true_positions_valid = true_positions[valid_mask]
                pred_positions_valid = pred_positions[valid_mask]
                
                # MAE, MSE 계산
                val_regression_mae = np.mean(np.abs(true_positions_valid - pred_positions_valid))
                val_regression_mse = np.mean((true_positions_valid - pred_positions_valid) ** 2)
                
                # 파단 위치 예측 오차 상세 분석 (height, degree 별 오차)
                height_error = np.abs(true_positions_valid[:, 0] - pred_positions_valid[:, 0])
                degree_error = np.abs(true_positions_valid[:, 1] - pred_positions_valid[:, 1])
                
                regression_error_details = {
                    'n_samples': int(np.sum(valid_mask)),
                    'height_error': {
                        'mean': float(np.mean(height_error)),
                        'std': float(np.std(height_error)),
                        'median': float(np.median(height_error)),
                        'min': float(np.min(height_error)),
                        'max': float(np.max(height_error)),
                        'q25': float(np.percentile(height_error, 25)),
                        'q75': float(np.percentile(height_error, 75))
                    },
                    'degree_error': {
                        'mean': float(np.mean(degree_error)),
                        'std': float(np.std(degree_error)),
                        'median': float(np.median(degree_error)),
                        'min': float(np.min(degree_error)),
                        'max': float(np.max(degree_error)),
                        'q25': float(np.percentile(degree_error, 25)),
                        'q75': float(np.percentile(degree_error, 75))
                    },
                    'combined_mae': float(val_regression_mae),
                    'combined_mse': float(val_regression_mse),
                    'combined_rmse': float(np.sqrt(val_regression_mse))
                }
                
                print(f"\n회귀 성능 (파단 샘플 {np.sum(valid_mask)}개):")
                print(f"  MAE: {val_regression_mae:.4f}")
                print(f"  MSE: {val_regression_mse:.4f}")
                print(f"  RMSE: {np.sqrt(val_regression_mse):.4f}")
                print(f"  Height 오차: 평균={np.mean(height_error):.4f}, 표준편차={np.std(height_error):.4f}")
                print(f"  Degree 오차: 평균={np.mean(degree_error):.4f}, 표준편차={np.std(degree_error):.4f}")
    
    # 최적 에폭 찾기 (검증 정확도 기준)
    best_epoch = None
    if monitor_metric in history.history:
        best_epoch = int(np.argmax(history.history[monitor_metric])) + 1  # 1-based indexing
        best_val_accuracy = float(np.max(history.history[monitor_metric]))
    
    # 데이터 분포 정보
    train_class_distribution = {
        'normal': int(np.sum(y_train == 0)),
        'break': int(np.sum(y_train == 1)),
        'total': int(len(y_train))
    }
    val_class_distribution = {
        'normal': int(np.sum(y_val == 0)),
        'break': int(np.sum(y_val == 1)),
        'total': int(len(y_val))
    }
    
    # 학습 정보 (상세 기록)
    train_info = {
        'model_type': 'resnet_1d_multi_output',
        'training_time': {
            'start': train_start_datetime,
            'end': train_end_datetime,
            'duration_seconds': float(train_duration),
            'duration_hours': float(train_duration_hours)
        },
        'model_architecture': {
            'sequence_length': int(sequence_length),
            'n_features': int(n_features),
            'num_filters': num_filters,
            'num_blocks': num_blocks,
            'blocks_per_layer': blocks_per_layer,
            'kernel_size': kernel_size,
            'dropout_rate': dropout_rate,
            'regularizer': float(regularizer) if regularizer else None
        },
        'training_hyperparameters': {
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'max_epochs': int(epochs),
            'epochs_trained': int(len(history.history['loss'])),
            'best_epoch': best_epoch,
            'best_val_accuracy': float(best_val_accuracy) if best_epoch else None,
            'regression_weight': float(regression_weight),
            'class_weight': {int(k): float(v) for k, v in class_weight.items()} if class_weight else None,
            'use_class_weight': use_class_weight,
            'optimal_threshold': float(threshold)
        },
        'data_info': {
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'train_class_distribution': train_class_distribution,
            'val_class_distribution': val_class_distribution,
            'class_imbalance_ratio': float(train_class_distribution['normal'] / train_class_distribution['break']) if train_class_distribution['break'] > 0 else None
        },
        'classification_performance': {
            'accuracy': float(val_accuracy),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1_score': float(val_f1),
            'auc': float(val_auc) if val_auc else None,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'classification_report': classification_report_dict,
            'roc_curve': roc_curve_data,
            'precision_recall_curve': pr_curve_data
        },
        'regression_performance': {
            'mae': float(val_regression_mae) if val_regression_mae is not None else None,
            'mse': float(val_regression_mse) if val_regression_mse is not None else None,
            'rmse': float(np.sqrt(val_regression_mse)) if val_regression_mse is not None else None,
            'error_details': regression_error_details
        },
        'history': {
            'loss': [float(v) for v in history.history['loss']],
            'val_loss': [float(v) for v in history.history['val_loss']],
        }
    }
    
    # 히스토리에서 분류 및 회귀 메트릭 추가
    for key in history.history.keys():
        if key not in ['loss', 'val_loss']:
            train_info['history'][key] = [float(v) for v in history.history[key]]
    
    if metadata:
        train_info['data_metadata'] = {
            'use_sliding_window': metadata.get('use_sliding_window', False),
            'window_height': metadata.get('window_height'),
            'window_degree': metadata.get('window_degree'),
            'stride_height': metadata.get('stride_height'),
            'stride_degree': metadata.get('stride_degree'),
        }
    
    # 모델 정보 저장
    info_file = output_path / "break_pattern_resnet_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    
    # 학습 곡선 시각화 및 상세 분석 시각화
    plot_history(train_info['history'], output_path)
    plot_detailed_analysis(
        y_val, y_pred, y_pred_proba, 
        positions_val, y_pred_positions,
        output_path, train_info
    )
    
    # 결과 해석 및 분석
    print(f"\n{'='*80}")
    print(f"검증 성능 분석")
    print(f"{'='*80}")
    print(f"\n[분류 성능]")
    print(f"  정확도: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  정밀도: {val_precision:.4f} ({val_precision*100:.2f}%)")
    print(f"  재현율: {val_recall:.4f} ({val_recall*100:.2f}%)")
    print(f"  F1 스코어: {val_f1:.4f}")
    if val_auc:
        print(f"  AUC: {val_auc:.4f}")
    
    print(f"\n[혼동 행렬 분석]")
    print(f"  TN (True Negative): {tn} - 정상을 정상으로 정확히 예측")
    print(f"  FP (False Positive): {fp} - 정상을 파단으로 오분류")
    print(f"  FN (False Negative): {fn} - 파단을 정상으로 오분류 ⚠️ (감소 필요)")
    print(f"  TP (True Positive): {tp} - 파단을 파단으로 정확히 예측")
    
    # False Negative 비율 계산
    total_break = fn + tp
    if total_break > 0:
        fn_rate = fn / total_break
        print(f"\n  False Negative 비율: {fn_rate:.4f} ({fn_rate*100:.2f}%)")
        print(f"    → 실제 파단 {total_break}개 중 {fn}개를 정상으로 오분류")
        if fn_rate > 0.3:
            print(f"    ⚠️ 경고: False Negative 비율이 30%를 초과합니다. 개선이 필요합니다.")
        elif fn_rate > 0.2:
            print(f"    ⚠️ 주의: False Negative 비율이 20%를 초과합니다.")
        else:
            print(f"    ✓ 양호: False Negative 비율이 20% 이하입니다.")
    
    # 회귀 성능 분석
    if positions_val is not None and y_pred_positions is not None:
        print(f"\n[회귀 성능 분석] (파단 위치 예측)")
        if val_regression_mae is not None:
            print(f"  전체 MAE: {val_regression_mae:.4f}")
            print(f"  전체 RMSE: {np.sqrt(val_regression_mse):.4f}")
            if regression_error_details:
                height_mae = regression_error_details['height_error']['mean']
                degree_mae = regression_error_details['degree_error']['mean']
                print(f"  Height MAE: {height_mae:.4f}")
                print(f"  Degree MAE: {degree_mae:.4f}")
                
                # 각도 예측이 높이 예측보다 나쁜지 확인
                if degree_mae > height_mae * 1.1:
                    print(f"    ⚠️ 주의: 각도 예측 오차가 높이 예측 오차보다 큽니다.")
                    print(f"    → 각도 예측 정확도 개선이 필요합니다.")
                else:
                    print(f"    ✓ 양호: 각도 예측과 높이 예측이 균형잡혀 있습니다.")
    
    print(f"\n[학습 정보]")
    print(f"  학습 시간: {train_duration_hours:.2f}시간 ({train_duration:.0f}초)")
    print(f"  시작: {train_start_datetime}")
    print(f"  종료: {train_end_datetime}")
    if best_epoch:
        print(f"  최적 에폭: {best_epoch} (검증 정확도: {best_val_accuracy:.4f})")
    
    print(f"\n{'='*80}")
    
    print(f"\n모델 저장 완료 (날짜 폴더: {date_folder}):")
    print(f"  모델 파일: {model_file}")
    print(f"  정보 파일: {info_file}")
    print(f"  학습 곡선: {output_path / 'break_pattern_resnet_training_history.png'}")
    print(f"  상세 분석: {output_path / 'break_pattern_resnet_detailed_analysis.png'}")
    
    return model, train_info, threshold


def plot_history(history, output_path: Path):
    """
    학습 곡선 시각화.
    
    Args:
        history: 학습 히스토리
        output_path: 출력 디렉토리
    """
    # 다중 출력 모델인지 확인
    is_multi_output = any('classification' in key or 'regression' in key for key in history.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    if 'loss' in history:
        axes[0, 0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy (단일 출력 또는 다중 출력)
    if is_multi_output:
        if 'classification_accuracy' in history:
            axes[0, 1].plot(history['classification_accuracy'], label='Train Accuracy')
        if 'val_classification_accuracy' in history:
            axes[0, 1].plot(history['val_classification_accuracy'], label='Val Accuracy')
    else:
        if 'accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision (단일 출력 또는 다중 출력)
    if is_multi_output:
        if 'classification_precision' in history:
            axes[1, 0].plot(history['classification_precision'], label='Train Precision')
        if 'val_classification_precision' in history:
            axes[1, 0].plot(history['val_classification_precision'], label='Val Precision')
    else:
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
        if 'val_precision' in history:
            axes[1, 0].plot(history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall (단일 출력 또는 다중 출력)
    if is_multi_output:
        if 'classification_recall' in history:
            axes[1, 1].plot(history['classification_recall'], label='Train Recall')
        if 'val_classification_recall' in history:
            axes[1, 1].plot(history['val_classification_recall'], label='Val Recall')
    else:
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
        if 'val_recall' in history:
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    plot_file = output_path / "break_pattern_resnet_training_history.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"학습 곡선 저장 완료: {plot_file}")


def plot_detailed_analysis(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    positions_val: Optional[np.ndarray],
    y_pred_positions: Optional[np.ndarray],
    output_path: Path,
    train_info: Dict
):
    """
    상세 분석 시각화 (혼동 행렬, ROC 곡선, PR 곡선, 회귀 오차 분포 등).
    
    Args:
        y_val: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률
        positions_val: 실제 파단 위치
        y_pred_positions: 예측 파단 위치
        output_path: 출력 디렉토리
        train_info: 학습 정보 딕셔너리
    """
    from matplotlib import cm
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 혼동 행렬
    ax1 = fig.add_subplot(gs[0, 0])
    cm_matrix = confusion_matrix(y_val, y_pred)
    im = ax1.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    
    classes = ['정상 (Normal)', '파단 (Break)']
    ax1.set(xticks=np.arange(cm_matrix.shape[1]),
            yticks=np.arange(cm_matrix.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title='Confusion Matrix',
            ylabel='실제 (True Label)',
            xlabel='예측 (Predicted Label)')
    
    # 셀에 값 표시
    thresh = cm_matrix.max() / 2.
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            ax1.text(j, i, format(cm_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_matrix[i, j] > thresh else "black")
    
    # 2. ROC 곡선
    ax2 = fig.add_subplot(gs[0, 1])
    if train_info['classification_performance']['roc_curve']:
        roc_data = train_info['classification_performance']['roc_curve']
        ax2.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC curve (AUC = {train_info["classification_performance"]["auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True)
    
    # 3. Precision-Recall 곡선
    ax3 = fig.add_subplot(gs[0, 2])
    if train_info['classification_performance']['precision_recall_curve']:
        pr_data = train_info['classification_performance']['precision_recall_curve']
        ax3.plot(pr_data['recall'], pr_data['precision'])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.grid(True)
    
    # 4. 예측 확률 분포
    ax4 = fig.add_subplot(gs[1, 0])
    normal_proba = y_pred_proba[y_val == 0]
    break_proba = y_pred_proba[y_val == 1]
    ax4.hist(normal_proba, bins=30, alpha=0.5, label='정상 (Normal)', color='blue')
    ax4.hist(break_proba, bins=30, alpha=0.5, label='파단 (Break)', color='red')
    ax4.axvline(x=train_info['training_hyperparameters']['optimal_threshold'], 
               color='green', linestyle='--', label=f'임계값 = {train_info["training_hyperparameters"]["optimal_threshold"]:.3f}')
    ax4.set_xlabel('예측 확률')
    ax4.set_ylabel('빈도')
    ax4.set_title('예측 확률 분포')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 회귀 오차 분포 (Height)
    ax5 = fig.add_subplot(gs[1, 1])
    if train_info['regression_performance']['error_details']:
        error_details = train_info['regression_performance']['error_details']
        # 실제 데이터는 이미 계산되어 있지만, 시각화를 위해 다시 계산
        if positions_val is not None and y_pred_positions is not None:
            break_mask = (y_val == 1)
            if np.any(break_mask):
                break_indices = np.where(break_mask)[0]
                true_positions = positions_val[break_indices]
                pred_positions = y_pred_positions[break_indices]
                valid_mask = np.any(true_positions > 0, axis=1)
                if np.any(valid_mask):
                    true_positions_valid = true_positions[valid_mask]
                    pred_positions_valid = pred_positions[valid_mask]
                    height_error = np.abs(true_positions_valid[:, 0] - pred_positions_valid[:, 0])
                    ax5.hist(height_error, bins=30, alpha=0.7, color='orange')
                    ax5.axvline(x=error_details['height_error']['mean'], 
                               color='red', linestyle='--', 
                               label=f'평균 = {error_details["height_error"]["mean"]:.4f}')
                    ax5.set_xlabel('Height 오차')
                    ax5.set_ylabel('빈도')
                    ax5.set_title('Height 예측 오차 분포')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
    
    # 6. 회귀 오차 분포 (Degree)
    ax6 = fig.add_subplot(gs[1, 2])
    if train_info['regression_performance']['error_details']:
        error_details = train_info['regression_performance']['error_details']
        if positions_val is not None and y_pred_positions is not None:
            break_mask = (y_val == 1)
            if np.any(break_mask):
                break_indices = np.where(break_mask)[0]
                true_positions = positions_val[break_indices]
                pred_positions = y_pred_positions[break_indices]
                valid_mask = np.any(true_positions > 0, axis=1)
                if np.any(valid_mask):
                    true_positions_valid = true_positions[valid_mask]
                    pred_positions_valid = pred_positions[valid_mask]
                    degree_error = np.abs(true_positions_valid[:, 1] - pred_positions_valid[:, 1])
                    ax6.hist(degree_error, bins=30, alpha=0.7, color='purple')
                    ax6.axvline(x=error_details['degree_error']['mean'], 
                               color='red', linestyle='--', 
                               label=f'평균 = {error_details["degree_error"]["mean"]:.4f}')
                    ax6.set_xlabel('Degree 오차')
                    ax6.set_ylabel('빈도')
                    ax6.set_title('Degree 예측 오차 분포')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
    
    # 7. 학습 시간 정보
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    time_info = train_info['training_time']
    text = f"""학습 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
시작 시간: {time_info['start']}
종료 시간: {time_info['end']}
학습 시간: {time_info['duration_hours']:.2f}시간
최적 에폭: {train_info['training_hyperparameters'].get('best_epoch', 'N/A')}
"""
    ax7.text(0.1, 0.5, text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    # 8. 분류 성능 요약
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    perf = train_info['classification_performance']
    # AUC 값 포맷팅 (None 체크)
    auc_str = f"{perf['auc']:.4f}" if perf['auc'] is not None else 'N/A'
    text = f"""분류 성능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
정확도: {perf['accuracy']:.4f}
정밀도: {perf['precision']:.4f}
재현율: {perf['recall']:.4f}
F1 스코어: {perf['f1_score']:.4f}
AUC: {auc_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
혼동 행렬
TN: {perf['confusion_matrix']['tn']}, FP: {perf['confusion_matrix']['fp']}
FN: {perf['confusion_matrix']['fn']}, TP: {perf['confusion_matrix']['tp']}
"""
    ax8.text(0.1, 0.5, text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax8.transAxes)
    
    # 9. 회귀 성능 요약
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    reg_perf = train_info['regression_performance']
    if reg_perf['error_details']:
        error_details = reg_perf['error_details']
        text = f"""회귀 성능 (파단 위치 예측)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
샘플 수: {error_details['n_samples']}개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 오차
MAE: {reg_perf['mae']:.4f}
MSE: {reg_perf['mse']:.4f}
RMSE: {reg_perf['rmse']:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Height 오차
평균: {error_details['height_error']['mean']:.4f}
중앙값: {error_details['height_error']['median']:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Degree 오차
평균: {error_details['degree_error']['mean']:.4f}
중앙값: {error_details['degree_error']['median']:.4f}
"""
    else:
        text = "회귀 성능 데이터 없음"
    ax9.text(0.1, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax9.transAxes)
    
    plt.suptitle('상세 학습 결과 분석', fontsize=16, y=0.995)
    
    plot_file = output_path / "break_pattern_resnet_detailed_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"상세 분석 시각화 저장 완료: {plot_file}")


def main():
    # 기본 설정값
    sequences = "5. train_data/train/break_sequences_train.npy"
    labels = "5. train_data/train/break_labels_train.npy"
    metadata_path = "5. train_data/break_sequences_metadata.json"
    test_sequences = "5. train_data/test/break_sequences_test.npy"
    test_labels = "5. train_data/test/break_labels_test.npy"
    # window_positions 또는 break_positions (슬라이딩 윈도우 모드에서는 window_positions 사용)
    # 먼저 window_positions 시도
    positions = "5. train_data/train/window_positions_train.npy"
    test_positions = "5. train_data/test/window_positions_test.npy"
    # window_positions가 없으면 break_positions 시도
    positions_full_path = os.path.join(current_dir, positions)
    if not os.path.exists(positions_full_path):
        positions = "5. train_data/train/break_positions_train.npy"
        test_positions = "5. train_data/test/break_positions_test.npy"
    regression_weight = 1.0  # 회귀 loss 가중치 (실제 가중치는 이 값의 7배 = 7.0, 파단 위치 예측 정확도 향상)
    use_train_test_split = False
    output_dir = "6. models"
    num_filters = 128  # 64 -> 128: 모델 용량 증가로 표현력 향상
    num_blocks = 4  # 3 -> 4: 더 깊은 네트워크로 복잡한 패턴 학습
    blocks_per_layer = 2
    kernel_size = 7
    dropout = 0.4  # 0.3 -> 0.4: 과적합 방지 강화 (회귀 오차가 큰 것 방지)
    learning_rate = 0.0005  # 0.001 -> 0.0005: 더 안정적인 학습
    batch_size = 128  # 64 -> 128: 배치 크기 증가로 학습 속도 향상 (약 2배 빠름)
    epochs = 150  # 100 -> 150: 더 긴 학습 허용
    regularizer = 1e-4  # 1e-5 -> 1e-4: 정규화 강화로 과적합 방지
    test_size = 0.2
    random_state = 42
    use_class_weight = True
    optimal_threshold = True
    
    print("=" * 80)
    print("ResNet 모델 학습 시작")
    print("=" * 80)
    
    # 데이터 로드
    if not use_train_test_split:
        # 이미 분할된 train/test 데이터 사용 (기본값)
        print("이미 분할된 train/test 데이터 사용")
        X_train, y_train, positions_train, metadata_train = load_sequence_data(
            sequences,
            labels,
            metadata_path,
            positions
        )
        X_val, y_val, positions_val, metadata_val = load_sequence_data(
            test_sequences,
            test_labels,
            None,
            test_positions
        )
        metadata = metadata_train  # 메타데이터는 train에서 가져옴
    else:
        # 전체 데이터 로드 후 분할
        X, y, positions_data, metadata = load_sequence_data(
            sequences,
            labels,
            metadata_path,
            positions
        )
        
        # 학습/검증 데이터 분할
        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        # 파단 위치 데이터도 동일한 인덱스로 분할
        if positions_data is not None:
            positions_train = positions_data[train_indices]
            positions_val = positions_data[val_indices]
        else:
            positions_train = None
            positions_val = None
    
    print(f"\n데이터 분할 완료:")
    print(f"  학습 세트: {len(X_train)}개")
    print(f"  검증 세트: {len(X_val)}개")
    
    # 모델 학습
    model, train_info, threshold = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        positions_train=positions_train,
        positions_val=positions_val,
        num_filters=num_filters,
        num_blocks=num_blocks,
        blocks_per_layer=blocks_per_layer,
        kernel_size=kernel_size,
        dropout_rate=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        regularizer=regularizer,
        output_dir=output_dir,
        use_class_weight=use_class_weight,
        optimal_threshold=optimal_threshold,
        regression_weight=regression_weight,
        metadata=metadata
    )
    
    # 학습 곡선 시각화는 train_model 함수 내부에서 이미 수행됨
    # (날짜 폴더에 저장되도록 수정됨)
    
    print("\n" + "=" * 80)
    print("ResNet 모델 학습 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
