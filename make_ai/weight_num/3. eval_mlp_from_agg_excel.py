#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
규칙 대신 소형 MLP로 (light, max_x, max_z) -> 파단 예측 학습.
동일 엑셀(export_2nd_eval_to_excel)에서 특징 추출 후 학습, 3구간(정상/보류/파단) 적용.

3구간:
  - 정상: MLP 확률 < 0.65
  - 보류: 0.65 <= MLP 확률 < t_high (t_high = precision>=0.9 인 최소 threshold)
  - 파단: MLP 확률 >= t_high (precision 0.9 이상 되도록 설정)

사용 순서:
  1) 1. export_2nd_eval_to_excel.py 로 엑셀 생성
  2) python "3. eval_mlp_from_agg_excel.py"

출력:
  - eval_mlp_model.keras (학습된 모델)
  - eval_mlp_3zone_thresholds.txt (정상_상한, 파단_하한)
  - 콘솔: threshold 스윕 결과 + 3구간 요약
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


def extract_max_score_from_boxes(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    scores = []
    for m in re.finditer(r"score=([0-9]*\.?[0-9]+)", text):
        try:
            scores.append(float(m.group(1)))
        except ValueError:
            continue
    return float(max(scores)) if scores else 0.0


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(recall=recall, precision=precision, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    if not HAS_TF:
        print("TensorFlow가 필요합니다: pip install tensorflow")
        return

    base_dir = Path(__file__).resolve().parent
    xlsx_path = base_dir / "export_2nd_eval_to_excel" / "hard_2nd_merge_data_predictions.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {xlsx_path}\n먼저 1. export_2nd_eval_to_excel.py 를 실행하세요.")

    df = pd.read_excel(xlsx_path)
    for c in ["label", "x_boxes", "y_boxes", "z_boxes"]:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {c}")

    y = df["label"].astype(int).values
    df["max_score_x"] = df["x_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_y"] = df["y_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_z"] = df["z_boxes"].astype(str).apply(extract_max_score_from_boxes)

    # 특징: light(있으면), max_x, max_y, max_z
    use_light = "light_prob_break" in df.columns
    if use_light:
        X = np.column_stack([
            df["light_prob_break"].astype(float).fillna(0.0).values,
            df["max_score_x"].values.astype(float),
            df["max_score_y"].values.astype(float),
            df["max_score_z"].values.astype(float),
        ])
    else:
        X = np.column_stack([
            df["max_score_x"].values.astype(float),
            df["max_score_y"].values.astype(float),
            df["max_score_z"].values.astype(float),
        ])

    X = X.astype(np.float32)
    y_flat = y.astype(np.int32)
    y = y_flat.astype(np.float32).reshape(-1, 1)

    n_pos = int(np.sum(y_flat == 1))
    n_neg = int(np.sum(y_flat == 0))
    print(f"샘플 수: {len(y)}, 양성(파단): {n_pos}, 음성: {n_neg}")

    # 양성 오버샘플링: 소수 클래스 반복하여 학습 시 양성 비율 확대 (recall+precision 목표에 유리)
    idx_pos = np.where(y_flat == 1)[0]
    idx_neg = np.where(y_flat == 0)[0]
    repeat_pos = max(1, min(10, n_neg // max(1, n_pos)))  # 양성 약 5~8배 반복
    X_train = np.vstack([X] + [X[idx_pos]] * (repeat_pos - 1))
    y_train = np.vstack([y] + [y[idx_pos]] * (repeat_pos - 1))
    # 셔플
    perm = np.random.default_rng(42).permutation(len(y_train))
    X_train = X_train[perm].astype(np.float32)
    y_train = y_train[perm]
    print(f"학습 시 오버샘플링: 양성 {repeat_pos}배 반복 -> {len(y_train)} 샘플")

    keras.utils.set_random_seed(42)
    n_in = X.shape[1]
    model = keras.Sequential([
        keras.layers.Input((n_in,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.08),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        verbose=0,
    )

    # 원본 데이터 기준 예측 (평가는 원본 라벨로)
    proba = model.predict(X, verbose=0).ravel()

    # threshold 스윕: 0.05~0.95, recall+precision 최대 구간 확인
    all_results = []
    for th in np.linspace(0.05, 0.95, 37):
        pred = (proba >= th).astype(int)
        m = compute_metrics(y_flat, pred)
        m["threshold"] = float(th)
        m["recall_plus_precision"] = m["recall"] + m["precision"]
        all_results.append(m)

    best_overall = max(all_results, key=lambda x: (x["recall_plus_precision"], x["f1"]))
    print(f"\n전체 구간 최고 r+p: t={best_overall['threshold']:.2f}  recall={best_overall['recall']:.3f}, precision={best_overall['precision']:.3f}, r+p={best_overall['recall_plus_precision']:.3f}")

    # recall + precision >= 1 인 것만 출력
    results = [r for r in all_results if r["recall_plus_precision"] >= 1.0]
    results.sort(key=lambda x: (-x["recall_plus_precision"], -x["f1"]))
    print("\n=== MLP 임계값 스윕 (recall + precision >= 1 인 결과만) ===")
    for r in results[:25]:
        print(
            f"  t={r['threshold']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, "
            f"r+p={r['recall_plus_precision']:.3f}, F1={r['f1']:.3f}"
        )
    if not results:
        print("  (recall+precision >= 1 인 구간 없음) — 상위 5개 threshold:")
        for r in sorted(all_results, key=lambda x: (-x["recall_plus_precision"], -x["f1"]))[:5]:
            print(f"  t={r['threshold']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, r+p={r['recall_plus_precision']:.3f}")

    if results:
        best = results[0]
        print(f"\n추천: threshold={best['threshold']:.2f}  recall={best['recall']:.3f}, precision={best['precision']:.3f}, r+p={best['recall_plus_precision']:.3f}")

    # ---- 3구간: 정상 / 보류 / 파단 ----
    # 정상: MLP < 0.65
    # 보류: 0.65 <= MLP < t_high (precision>=0.9 인 파단 기준선)
    # 파단: MLP >= t_high (precision >= 0.9 되도록 하는 최소 threshold = 보류 상한)
    THRESH_NORMAL = 0.65  # 이 미만 = 정상
    PRECISION_PARDAN_MIN = 0.9  # 파단 구간 precision 하한

    # precision >= 0.9 인 threshold 중 최소값(가장 낮은 기준선) = 파단으로 쓸 때 precision 0.9 달성
    candidates_high = [r for r in all_results if r["precision"] >= PRECISION_PARDAN_MIN and r["threshold"] >= THRESH_NORMAL]
    if candidates_high:
        # threshold 낮은 순으로 정렬 후 가장 낮은 것 = recall 최대화하면서 precision 0.9
        best_high = min(candidates_high, key=lambda x: x["threshold"])
        thresh_pardan = best_high["threshold"]
    else:
        # precision 0.9 달성 불가 시 가장 높은 precision 주는 threshold 사용
        above_65 = [r for r in all_results if r["threshold"] >= THRESH_NORMAL]
        thresh_pardan = max(above_65, key=lambda x: x["precision"])["threshold"] if above_65 else 0.95

    # 구간별 판정
    pred_normal = proba < THRESH_NORMAL
    pred_hold = (proba >= THRESH_NORMAL) & (proba < thresh_pardan)
    pred_pardan = proba >= thresh_pardan

    n_normal = int(np.sum(pred_normal))
    n_hold = int(np.sum(pred_hold))
    n_pardan = int(np.sum(pred_pardan))
    # 파단 구간의 precision (실제 파단인 것 중 파단으로 부른 비율이 아니라, 파단으로 부른 것 중 실제 파단 비율)
    tp_p = int(np.sum((y_flat == 1) & pred_pardan))
    fp_p = int(np.sum((y_flat == 0) & pred_pardan))
    prec_pardan = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
    rec_pardan = tp_p / n_pos if n_pos > 0 else 0.0

    print("\n" + "=" * 60)
    print("3구간: 정상 / 보류 / 파단")
    print("=" * 60)
    print(f"  정상: MLP < {THRESH_NORMAL:.2f}  -> {n_normal}건")
    print(f"  보류: {THRESH_NORMAL:.2f} <= MLP < {thresh_pardan:.2f}  -> {n_hold}건")
    print(f"  파단: MLP >= {thresh_pardan:.2f}  -> {n_pardan}건  (precision={prec_pardan:.3f}, recall={rec_pardan:.3f})")
    print("=" * 60)

    out_dir = base_dir / "export_2nd_eval_to_excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "eval_mlp_model.keras"
    model.save(model_path)
    print(f"\n모델 저장: {model_path}")

    # 3구간 threshold 저장 (다른 스크립트에서 사용 가능)
    thresholds_path = out_dir / "eval_mlp_3zone_thresholds.txt"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        f.write(f"# 3구간: 정상 / 보류 / 파단\n")
        f.write(f"정상_상한\t{THRESH_NORMAL}\n")
        f.write(f"파단_하한\t{thresh_pardan}\n")
        f.write(f"# 정상: prob < 정상_상한, 보류: 정상_상한 <= prob < 파단_하한, 파단: prob >= 파단_하한\n")
    print(f"3구간 threshold 저장: {thresholds_path}")


if __name__ == "__main__":
    main()
