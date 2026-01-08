#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파단 검출 모델 학습 스크립트.

입력: train_data/break_features.csv (피처 벡터)
출력: models/break_detection_model.pkl (학습된 모델)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_training_data(
    break_features_path: str,
    normal_features_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    학습 데이터 로드.
    
    Args:
        break_features_path: 파단 피처 CSV 경로
        normal_features_path: 정상 피처 CSV 경로 (선택사항)
    
    Returns:
        tuple: (X, y) 피처와 라벨
    """
    # 파단 데이터 로드
    df_break = pd.read_csv(break_features_path)
    
    if normal_features_path and os.path.exists(normal_features_path):
        # 정상 데이터도 있는 경우
        df_normal = pd.read_csv(normal_features_path)
        df = pd.concat([df_break, df_normal], ignore_index=True)
        print(f"파단 샘플: {len(df_break)}개, 정상 샘플: {len(df_normal)}개")
    else:
        # 파단 데이터만 있는 경우
        df = df_break.copy()
        print(f"파단 샘플: {len(df)}개 (정상 샘플 없음 - 이상 탐지 모드)")
    
    # 피처 컬럼과 라벨 분리
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'poleid', 'project_name', 'csv_path']]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # 메타데이터 저장
    metadata = {
        'poleid': df['poleid'].tolist() if 'poleid' in df.columns else [],
        'project_name': df['project_name'].tolist() if 'project_name' in df.columns else [],
    }
    
    return X, y, feature_cols, metadata


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = 'random_forest',
    use_scaling: bool = True,
    **kwargs
) -> Tuple[object, object, dict]:
    """
    모델 학습.
    
    Args:
        X_train: 학습 피처
        y_train: 학습 라벨
        X_val: 검증 피처
        y_val: 검증 라벨
        model_type: 모델 타입 ('random_forest', 'gradient_boosting', 'logistic', 'svm')
        use_scaling: 스케일링 사용 여부
        **kwargs: 모델별 하이퍼파라미터
    
    Returns:
        tuple: (모델, 스케일러, 학습 정보)
    """
    # 스케일링
    scaler = None
    if use_scaling:
        scaler = RobustScaler()  # 이상치에 강건한 스케일러
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train.values
        X_val_scaled = X_val.values
    
    # 모델 선택
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if len(np.unique(y_train)) > 1 else None
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=42
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            random_state=42,
            class_weight='balanced' if len(np.unique(y_train)) > 1 else None
        )
    elif model_type == 'svm':
        model = SVC(
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            probability=True,
            random_state=42,
            class_weight='balanced' if len(np.unique(y_train)) > 1 else None
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 모델 학습
    print(f"\n[{model_type}] 모델 학습 중...")
    model.fit(X_train_scaled, y_train)
    
    # 검증 성능 평가
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    train_info = {
        'model_type': model_type,
        'val_accuracy': accuracy_score(y_val, y_pred),
        'val_precision': precision_score(y_val, y_pred, zero_division=0),
        'val_recall': recall_score(y_val, y_pred, zero_division=0),
        'val_f1': f1_score(y_val, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None and len(np.unique(y_val)) > 1:
        train_info['val_roc_auc'] = roc_auc_score(y_val, y_pred_proba)
    
    return model, scaler, train_info


def evaluate_model(
    model: object,
    scaler: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str
) -> dict:
    """
    모델 평가 및 시각화.
    
    Args:
        model: 학습된 모델
        scaler: 스케일러
        X_test: 테스트 피처
        y_test: 테스트 라벨
        output_dir: 출력 디렉토리
    
    Returns:
        dict: 평가 결과
    """
    # 스케일링
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 평가 지표
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # 분류 보고서
    print("\n" + "=" * 80)
    print("테스트 세트 평가 결과")
    print("=" * 80)
    print(f"정확도 (Accuracy): {results['accuracy']:.4f}")
    print(f"정밀도 (Precision): {results['precision']:.4f}")
    print(f"재현율 (Recall): {results['recall']:.4f}")
    print(f"F1 점수: {results['f1_score']:.4f}")
    if 'roc_auc' in results:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=['정상', '파단']))
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n혼동 행렬:")
    print(cm)
    
    # 시각화
    output_path = Path(current_dir) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['정상', '파단'], yticklabels=['정상', '파단'])
    plt.title('Confusion Matrix')
    plt.ylabel('실제')
    plt.xlabel('예측')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # ROC 곡선 (이진 분류인 경우)
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curve.png', dpi=300)
        plt.close()
    
    # Precision-Recall 곡선
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300)
        plt.close()
    
    return results


def get_feature_importance(model: object, feature_names: list) -> pd.DataFrame:
    """
    피처 중요도 추출.
    
    Args:
        model: 학습된 모델
        feature_names: 피처 이름 리스트
    
    Returns:
        DataFrame: 피처 중요도
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df_importance


def main():
    parser = argparse.ArgumentParser(
        description="파단 검출 모델 학습"
    )
    parser.add_argument(
        "--break-features",
        type=str,
        default="train_data/break_features.csv",
        help="파단 피처 CSV 경로",
    )
    parser.add_argument(
        "--normal-features",
        type=str,
        default=None,
        help="정상 피처 CSV 경로 (선택사항)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=['random_forest', 'gradient_boosting', 'logistic', 'svm'],
        help="모델 타입",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="모델 저장 디렉토리",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="테스트 세트 비율",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="검증 세트 비율 (학습 세트에서 분리)",
    )
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        help="스케일링 사용 안 함",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="트리 모델의 트리 개수",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="트리 최대 깊이",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("파단 검출 모델 학습 시작")
    print("=" * 80)
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    X, y, feature_cols, metadata = load_training_data(
        args.break_features,
        args.normal_features
    )
    
    print(f"총 샘플 수: {len(X)}")
    print(f"피처 수: {len(feature_cols)}")
    print(f"클래스 분포: {y.value_counts().to_dict()}")
    
    # 데이터 분할
    # 먼저 테스트 세트 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # 학습/검증 세트 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=args.val_size, 
        random_state=42,
        stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None
    )
    
    print(f"\n데이터 분할:")
    print(f"  학습 세트: {len(X_train)}개")
    print(f"  검증 세트: {len(X_val)}개")
    print(f"  테스트 세트: {len(X_test)}개")
    
    # 모델 학습
    model, scaler, train_info = train_model(
        X_train, y_train, X_val, y_val,
        model_type=args.model_type,
        use_scaling=not args.no_scaling,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    print(f"\n검증 세트 성능:")
    for key, value in train_info.items():
        if key != 'model_type':
            print(f"  {key}: {value:.4f}")
    
    # 테스트 세트 평가
    print("\n테스트 세트 평가 중...")
    test_results = evaluate_model(
        model, scaler, X_test, y_test, args.output_dir
    )
    
    # 피처 중요도
    print("\n피처 중요도 추출 중...")
    df_importance = get_feature_importance(model, feature_cols)
    if df_importance is not None:
        output_path = Path(current_dir) / args.output_dir
        importance_path = output_path / 'feature_importance.csv'
        df_importance.to_csv(importance_path, index=False)
        print(f"피처 중요도 저장: {importance_path}")
        print("\n상위 10개 중요 피처:")
        print(df_importance.head(10).to_string(index=False))
        
        # 피처 중요도 시각화
        plt.figure(figsize=(10, 8))
        top_features = df_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 모델 저장
    output_path = Path(current_dir) / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / 'break_detection_model.pkl'
    scaler_path = output_path / 'scaler.pkl'
    
    joblib.dump(model, model_path)
    if scaler:
        joblib.dump(scaler, scaler_path)
    
    # 모델 정보 저장
    model_info = {
        'model_type': args.model_type,
        'feature_names': feature_cols,
        'num_features': len(feature_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_info': train_info,
        'test_results': test_results,
        'scaler_used': scaler is not None,
    }
    
    info_path = output_path / 'model_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n모델 저장 완료:")
    print(f"  모델: {model_path}")
    if scaler:
        print(f"  스케일러: {scaler_path}")
    print(f"  모델 정보: {info_path}")
    
    print("\n" + "=" * 80)
    print("모델 학습 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()

