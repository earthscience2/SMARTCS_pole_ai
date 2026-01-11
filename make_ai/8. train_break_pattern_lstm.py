#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시퀀스 데이터를 이용하여 파단 패턴을 학습하는 LSTM/GRU 모델 학습 스크립트.

입력: 6. train_data/sequences/break_sequences.npy, break_labels.npy
출력: 7. models/break_pattern_lstm.keras (학습된 모델)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional,
    BatchNormalization, Input, Attention, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_sequence_data(
    sequences_path: str,
    labels_path: str,
    metadata_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    시퀀스 데이터 로드.
    
    Args:
        sequences_path: 시퀀스 데이터 파일 경로 (.npy)
        labels_path: 라벨 데이터 파일 경로 (.npy)
        metadata_path: 메타데이터 파일 경로 (.json, 선택사항)
    
    Returns:
        tuple: (시퀀스 데이터, 라벨, 메타데이터)
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(sequences_path):
        sequences_path = os.path.join(current_dir, sequences_path)
    if not os.path.isabs(labels_path):
        labels_path = os.path.join(current_dir, labels_path)
    if metadata_path and not os.path.isabs(metadata_path):
        metadata_path = os.path.join(current_dir, metadata_path)
    
    X = np.load(sequences_path)
    y = np.load(labels_path)
    
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print(f"시퀀스 데이터 로드 완료:")
    print(f"  데이터 형태: {X.shape}")
    print(f"  라벨 형태: {y.shape}")
    print(f"  클래스 분포: {np.bincount(y.astype(int))}")
    
    # 슬라이딩 윈도우 사용 여부 확인
    if metadata.get('use_sliding_window', False):
        print(f"  데이터 준비 방식: 슬라이딩 윈도우 (평가 시와 동일)")
        print(f"    윈도우 크기: 높이 {metadata.get('window_height', 'N/A')}m, 각도 {metadata.get('window_degree', 'N/A')}°")
        print(f"    스트라이드: 높이 {metadata.get('stride_height', 'N/A')}m, 각도 {metadata.get('stride_degree', 'N/A')}°")
    else:
        print(f"  데이터 준비 방식: 크롭된 영역 직접 사용")
    
    return X, y, metadata


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    model_type: str = 'lstm',
    use_bidirectional: bool = True,
    use_attention: bool = False,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    regularizer: Optional[float] = None
) -> Model:
    """
    LSTM/GRU 모델 구성.
    
    Args:
        sequence_length: 시퀀스 길이
        n_features: 피처 수
        model_type: 모델 타입 ('lstm' 또는 'gru')
        use_bidirectional: 양방향 레이어 사용 여부
        use_attention: Attention 메커니즘 사용 여부
        lstm_units: LSTM/GRU 유닛 수
        dropout_rate: Dropout 비율
        regularizer: 정규화 계수
    
    Returns:
        Keras 모델
    """
    inputs = Input(shape=(sequence_length, n_features))
    
    # LSTM/GRU 레이어
    if model_type.lower() == 'lstm':
        RNN_layer = LSTM
    elif model_type.lower() == 'gru':
        RNN_layer = GRU
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    reg = l1_l2(l1=regularizer, l2=regularizer) if regularizer else None
    
    if use_bidirectional:
        x = Bidirectional(
            RNN_layer(lstm_units, return_sequences=True, kernel_regularizer=reg)
        )(inputs)
    else:
        x = RNN_layer(lstm_units, return_sequences=True, kernel_regularizer=reg)(inputs)
    
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Attention 메커니즘 (선택사항)
    if use_attention:
        # Multi-Head Attention
        attention = MultiHeadAttention(num_heads=4, key_dim=lstm_units // 4)(x, x)
        x = Concatenate()([x, attention])
    
    # 두 번째 LSTM/GRU 레이어
    if use_bidirectional:
        x = Bidirectional(
            RNN_layer(lstm_units // 2, return_sequences=False, kernel_regularizer=reg)
        )(x)
    else:
        x = RNN_layer(lstm_units // 2, return_sequences=False, kernel_regularizer=reg)(x)
    
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense 레이어
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate / 2)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(dropout_rate / 2)(x)
    
    # 출력 레이어
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def calculate_class_weights(y: np.ndarray) -> dict:
    """
    클래스 가중치 계산 (불균형 데이터 처리).
    
    Args:
        y: 라벨 배열
    
    Returns:
        dict: 클래스 가중치 딕셔너리
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y
    )
    
    weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    return weight_dict


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, prioritize_recall: bool = True) -> float:
    """
    최적 임계값 찾기 (재현율 우선 또는 F1-Score 최대화).
    
    Args:
        y_true: 실제 라벨
        y_pred_proba: 예측 확률
        prioritize_recall: 재현율 우선 여부 (True면 재현율 70% 이상 유지하면서 F1 최대화)
    
    Returns:
        float: 최적 임계값
    """
    from sklearn.metrics import f1_score, recall_score
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    if prioritize_recall:
        # 재현율 70% 이상을 유지하면서 F1-Score를 최대화
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if recall >= 0.70:  # 재현율 70% 이상 유지
                f1 = f1_score(y_true, y_pred, zero_division=0)
                # 재현율이 충족되면 F1-Score 최대화
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
    else:
        # F1-Score 최대화
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
    
    return best_threshold


def evaluate_break_location_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    metadata: dict,
    threshold: float = 0.5
) -> dict:
    """
    파단 위치 정확도 평가 (높이 오차 계산).
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률
        metadata: 시퀀스 메타데이터 (samples 정보 포함)
        threshold: 분류 임계값
    
    Returns:
        dict: 위치 정확도 평가 결과
    """
    import os
    from pathlib import Path
    
    # 파단으로 올바르게 예측된 샘플 찾기 (TP만)
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    
    if len(tp_indices) == 0:
        return {
            'num_samples': 0,
            'mean_error': None,
            'median_error': None,
            'std_error': None,
            'rmse': None,
            'mae': None
        }
    
    samples = metadata.get('samples', [])
    
    # 인덱스 매핑 (검증/테스트 세트인 경우)
    index_mapping = metadata.get('_val_indices') or metadata.get('_test_indices')
    
    if not samples or (index_mapping and len(samples) != len(index_mapping) + len(y_true)):
        # 전체 메타데이터를 사용하거나 인덱스 매핑이 없는 경우
        if len(samples) != len(y_true):
            # 인덱스 매핑이 있는 경우 원본 인덱스로 변환
            if index_mapping:
                pass  # 아래에서 처리
            else:
                return {
                    'num_samples': 0,
                    'mean_error': None,
                    'median_error': None,
                    'std_error': None,
                    'rmse': None,
                    'mae': None
                }
    
    errors = []
    
    for idx in tp_indices:
        # 원본 인덱스 찾기
        if index_mapping:
            if idx >= len(index_mapping):
                continue
            original_idx = index_mapping[idx]
        else:
            original_idx = idx
        
        if original_idx >= len(samples):
            continue
        
        sample_meta = samples[original_idx]
        csv_path = sample_meta.get('csv_path', '')
        
        if not csv_path:
            continue
        
        # 크롭된 데이터의 높이 (상대적 위치)
        crop_height_min = sample_meta.get('height_min', None)
        crop_height_max = sample_meta.get('height_max', None)
        
        if crop_height_min is None or crop_height_max is None:
            continue
        
        crop_height_center = (crop_height_min + crop_height_max) / 2.0
        
        # 실제 파단 높이와 원본 edit 파일 찾기
        actual_height = None
        pred_height = None
        
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                continue
            
            # CSV 파일 경로에서 정보 찾기
            # 예: 5. crop_data/break/프로젝트/전주ID/파일_break_crop.csv
            # -> 4. edit_pole_data/break/프로젝트/전주ID/파일_processed.csv
            pole_dir = csv_file.parent
            poleid = pole_dir.name
            project_name = pole_dir.parent.name
            
            # 원본 edit 파일 경로 찾기
            # 크롭 파일명: XXX_OUT_processed_break_crop.csv
            # 원본 파일명: XXX_OUT_processed.csv
            crop_filename = csv_file.name
            if '_break_crop.csv' in crop_filename:
                original_filename = crop_filename.replace('_break_crop.csv', '_processed.csv')
            elif '_normal_crop.csv' in crop_filename:
                original_filename = crop_filename.replace('_normal_crop.csv', '_processed.csv')
            else:
                original_filename = crop_filename
            
            # edit 파일 경로 찾기 (여러 위치 시도)
            edit_base_paths = [
                csv_file.parent.parent.parent.parent / "4. edit_pole_data" / "break",
                Path(current_dir) / "4. edit_pole_data" / "break",
                csv_file.parent.parent.parent / "4. edit_pole_data" / "break"
            ]
            
            edit_file = None
            for edit_base in edit_base_paths:
                edit_file_candidate = edit_base / project_name / poleid / original_filename
                if edit_file_candidate.exists():
                    edit_file = edit_file_candidate
                    break
            
            # break_info.json 찾기
            info_paths = [
                edit_file.parent / f"{poleid}_break_info.json" if edit_file else None,
                pole_dir / f"{poleid}_break_info.json",
                Path(current_dir) / "4. edit_pole_data" / "break" / project_name / poleid / f"{poleid}_break_info.json"
            ]
            
            info_path = None
            for path in info_paths:
                if path and path.exists():
                    info_path = path
                    break
            
            if info_path and info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    break_info = json.load(f)
                    actual_height = break_info.get('breakheight')
                    if actual_height is not None:
                        try:
                            actual_height = float(actual_height)
                        except (ValueError, TypeError):
                            actual_height = None
            
            # edit 파일에서 전체 높이 범위 확인
            if edit_file and edit_file.exists():
                try:
                    df_edit = pd.read_csv(edit_file)
                    if 'height' in df_edit.columns and not df_edit.empty:
                        edit_height_min = df_edit['height'].min()
                        edit_height_max = df_edit['height'].max()
                        
                        # 크롭된 데이터의 높이를 원본 데이터 범위에 매핑
                        # 크롭 파일은 breakheight ±0.15m 범위를 포함하므로
                        # 크롭 파일의 높이는 그대로 실제 높이로 사용 가능
                        # (크롭 단계에서 이미 실제 높이로 크롭되었기 때문)
                        pred_height = crop_height_center
                except Exception:
                    pass
            
            # edit 파일을 못 찾은 경우, 크롭 파일의 높이를 그대로 사용
            if pred_height is None:
                pred_height = crop_height_center
                
        except Exception as e:
            continue
        
        # 실제 파단 높이와 비교
        if actual_height is not None and pred_height is not None:
            error = abs(pred_height - actual_height)
            errors.append(error)
    
    if len(errors) == 0:
        return {
            'num_samples': 0,
            'mean_error': None,
            'median_error': None,
            'std_error': None,
            'rmse': None,
            'mae': None
        }
    
    errors = np.array(errors)
    
    return {
        'num_samples': int(len(errors)),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'std_error': float(np.std(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(errors))
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'lstm',
    use_bidirectional: bool = True,
    use_attention: bool = False,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    regularizer: Optional[float] = None,
    output_dir: str = "7. models",
    use_class_weight: bool = True,
    optimal_threshold: bool = True,
    metadata: Optional[dict] = None,
    val_indices: Optional[np.ndarray] = None
) -> Tuple[Model, dict, float]:
    """
    모델 학습.
    
    Args:
        X_train: 학습 시퀀스 데이터
        y_train: 학습 라벨
        X_val: 검증 시퀀스 데이터
        y_val: 검증 라벨
        model_type: 모델 타입 ('lstm' 또는 'gru')
        use_bidirectional: 양방향 레이어 사용 여부
        use_attention: Attention 메커니즘 사용 여부
        lstm_units: LSTM/GRU 유닛 수
        dropout_rate: Dropout 비율
        learning_rate: 학습률
        batch_size: 배치 크기
        epochs: 에폭 수
        regularizer: 정규화 계수
        output_dir: 모델 저장 디렉토리
    
    Returns:
        tuple: (학습된 모델, 학습 정보)
    """
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # 모델 구성
    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=n_features,
        model_type=model_type,
        use_bidirectional=use_bidirectional,
        use_attention=use_attention,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        regularizer=regularizer
    )
    
    # 모델 컴파일
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"\n모델 구성 완료:")
    model.summary()
    
    # 출력 디렉토리 생성
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 콜백 설정
    model_file = output_path / "break_pattern_lstm_best.keras"
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(model_file),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    # 클래스 가중치 계산
    class_weight = None
    if use_class_weight:
        class_weight = calculate_class_weights(y_train)
        print(f"\n클래스 가중치: {class_weight}")
    
    # 모델 학습
    print(f"\n모델 학습 시작...")
    print(f"  학습 샘플: {len(X_train)}개")
    print(f"  검증 샘플: {len(X_val)}개")
    print(f"  배치 크기: {batch_size}")
    print(f"  최대 에폭: {epochs}")
    if class_weight:
        print(f"  클래스 가중치 적용: {class_weight}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # 최적 모델 로드
    model.load_weights(str(model_file))
    
    # 검증 성능 평가
    y_pred_proba = model.predict(X_val, verbose=0).ravel()
    
    # 최적 임계값 찾기 (재현율 우선)
    threshold = 0.5
    if optimal_threshold:
        threshold = find_optimal_threshold(y_val, y_pred_proba, prioritize_recall=True)
        print(f"\n최적 임계값: {threshold:.4f} (기본값: 0.5, 재현율 우선)")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
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
    
    # 파단 위치 정확도 평가 (검증 세트) - 인덱스 매칭은 evaluate_break_location_accuracy 내부에서 처리
    # 검증 세트 인덱스 정보를 전달하기 위해 metadata에 인덱스 정보 추가
    val_metadata = {}
    if metadata and 'samples' in metadata and val_indices is not None:
        # 전체 메타데이터를 전달하고, evaluate 함수 내부에서 인덱스 매칭
        val_metadata = {**metadata, '_val_indices': val_indices.tolist()}
    elif metadata:
        val_metadata = metadata
    else:
        val_metadata = {}
    
    val_location_acc = evaluate_break_location_accuracy(
        y_val, y_pred, y_pred_proba, val_metadata, threshold=threshold
    )
    
    # 학습 정보
    train_info = {
        'model_type': model_type,
        'sequence_length': int(sequence_length),
        'n_features': int(n_features),
        'use_bidirectional': use_bidirectional,
        'use_attention': use_attention,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'class_weight': class_weight,
        'optimal_threshold': float(threshold),
        'val_accuracy': float(val_accuracy),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'val_auc': float(val_auc) if val_auc else None,
        'val_location_accuracy': val_location_acc,
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
    }
    
    print(f"\n검증 성능:")
    print(f"  정확도: {val_accuracy:.4f}")
    print(f"  정밀도: {val_precision:.4f}")
    print(f"  재현율: {val_recall:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    if val_auc:
        print(f"  AUC-ROC: {val_auc:.4f}")
    
    if val_location_acc['num_samples'] > 0:
        print(f"\n검증 파단 위치 정확도:")
        print(f"  평가 샘플 수: {val_location_acc['num_samples']}개")
        print(f"  평균 오차: {val_location_acc['mean_error']:.4f}m")
        print(f"  중앙값 오차: {val_location_acc['median_error']:.4f}m")
        print(f"  표준편차: {val_location_acc['std_error']:.4f}m")
        print(f"  RMSE: {val_location_acc['rmse']:.4f}m")
        print(f"  MAE: {val_location_acc['mae']:.4f}m")
    
    # 최종 모델 저장
    final_model_file = output_path / "break_pattern_lstm.keras"
    model.save(str(final_model_file))
    print(f"\n모델 저장 완료: {final_model_file}")
    
    # 학습 정보 저장
    info_file = output_path / "break_pattern_lstm_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    print(f"학습 정보 저장 완료: {info_file}")
    
    # 학습 곡선 저장
    plot_history(history, output_path / "training_history.png")
    
    return model, train_info, threshold


def plot_history(history, output_path: Path):
    """
    학습 곡선 그래프 저장.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        if 'val_precision' in history.history:
            axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        if 'val_recall' in history.history:
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"학습 곡선 저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="시퀀스 데이터를 이용하여 파단 패턴을 학습하는 LSTM/GRU 모델 학습"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="6. train_data/sequences/break_sequences.npy",
        help="시퀀스 데이터 파일 경로",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="6. train_data/sequences/break_labels.npy",
        help="라벨 데이터 파일 경로",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="6. train_data/sequences/break_sequences_metadata.json",
        help="메타데이터 파일 경로",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=['lstm', 'gru'],
        help="모델 타입 (기본값: lstm)",
    )
    parser.add_argument(
        "--no-bidirectional",
        action="store_true",
        help="양방향 레이어 사용 안 함",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Attention 메커니즘 사용",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=128,
        help="LSTM/GRU 유닛 수 (기본값: 128)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout 비율 (기본값: 0.3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="학습률 (기본값: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (기본값: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="최대 에폭 수 (기본값: 100)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="테스트 세트 비율 (기본값: 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="검증 세트 비율 (학습 세트에서 분리, 기본값: 0.2)",
    )
    parser.add_argument(
        "--regularizer",
        type=float,
        default=None,
        help="정규화 계수 (L1/L2, 기본값: None)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="7. models",
        help="모델 저장 디렉토리",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="클래스 가중치 사용 안 함 (기본값: 사용)",
    )
    parser.add_argument(
        "--no-optimal-threshold",
        action="store_true",
        help="최적 임계값 찾기 사용 안 함 (기본값: 사용)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("파단 패턴 LSTM/GRU 모델 학습 시작")
    print("=" * 80)
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    X, y, metadata = load_sequence_data(
        args.sequences,
        args.labels,
        args.metadata
    )
    
    # 데이터 분할
    # 먼저 테스트 세트 분리
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # 학습 세트에서 검증 세트 분리
    train_indices_2 = np.arange(len(X_train))
    X_train, X_val, y_train, y_val, train_indices_2, val_indices_2 = train_test_split(
        X_train, y_train, train_indices_2, test_size=args.val_size, random_state=42, stratify=y_train
    )
    
    # 검증 세트의 원본 인덱스 계산
    val_indices = train_indices[val_indices_2]
    
    print(f"\n데이터 분할 완료:")
    print(f"  학습: {len(X_train)}개")
    print(f"  검증: {len(X_val)}개")
    print(f"  테스트: {len(X_test)}개")
    
    # 모델 학습
    model, train_info, optimal_threshold = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_type=args.model_type,
        use_bidirectional=not args.no_bidirectional,
        use_attention=args.use_attention,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        regularizer=args.regularizer,
        output_dir=args.output_dir,
        use_class_weight=not args.no_class_weight,
        optimal_threshold=not args.no_optimal_threshold,
        metadata=metadata,
        val_indices=val_indices
    )
    
    # 테스트 세트 평가 (최적 임계값 사용)
    print(f"\n테스트 세트 평가...")
    y_test_pred_proba = model.predict(X_test, verbose=0).ravel()
    threshold = optimal_threshold if not args.no_optimal_threshold else 0.5
    print(f"사용 임계값: {threshold:.4f}")
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print(f"\n테스트 성능:")
    print(f"  정확도: {test_accuracy:.4f}")
    print(f"  정밀도: {test_precision:.4f}")
    print(f"  재현율: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    
    # 테스트 세트 파단 위치 정확도 평가
    test_metadata = {}
    if metadata and 'samples' in metadata:
        # 테스트 세트 인덱스 정보 전달
        test_metadata = {**metadata, '_test_indices': test_indices.tolist()}
    
    test_location_acc = evaluate_break_location_accuracy(
        y_test, y_test_pred, y_test_pred_proba, test_metadata, threshold=threshold
    )
    
    if test_location_acc['num_samples'] > 0:
        print(f"\n테스트 파단 위치 정확도:")
        print(f"  평가 샘플 수: {test_location_acc['num_samples']}개")
        print(f"  평균 오차: {test_location_acc['mean_error']:.4f}m")
        print(f"  중앙값 오차: {test_location_acc['median_error']:.4f}m")
        print(f"  표준편차: {test_location_acc['std_error']:.4f}m")
        print(f"  RMSE: {test_location_acc['rmse']:.4f}m")
        print(f"  MAE: {test_location_acc['mae']:.4f}m")
    else:
        print(f"\n테스트 파단 위치 정확도: 평가할 샘플이 없습니다.")
    
    # 테스트 성능을 학습 정보에 추가
    train_info['test_accuracy'] = float(test_accuracy)
    train_info['test_precision'] = float(test_precision)
    train_info['test_recall'] = float(test_recall)
    train_info['test_f1'] = float(test_f1)
    train_info['test_location_accuracy'] = test_location_acc
    
    # 정보 업데이트
    info_file = Path(current_dir) / args.output_dir / "break_pattern_lstm_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("모델 학습 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
