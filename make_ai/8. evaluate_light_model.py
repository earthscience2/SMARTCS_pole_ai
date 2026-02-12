"""Light 모델 상세 평가 - 테스트 데이터 기반 성능 평가 및 결과 시각화"""

import os
import sys
import json
import argparse
import importlib.util
import datetime
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_metrics

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="라이트 모델 평가")
parser.add_argument("--run", type=str, default=None, help="7. light_models 하위 run 이름")
parser.add_argument("--target-precision", type=float, default=0.80, help="재학습 기준 precision")
parser.add_argument("--target-recall", type=float, default=0.90, help="재학습 기준 recall")
parser.add_argument("--target-f1", type=float, default=0.84, help="재학습 기준 F1")
parser.add_argument(
    "--target-pass-mode",
    type=str,
    choices=["all_metrics", "recall_priority"],
    default="all_metrics",
    help="재학습 판정 모드",
)
parser.add_argument("--best-target-recall", type=float, default=0.90, help="best 모델 선별 기준: 최소 recall")
parser.add_argument("--best-target-accuracy", type=float, default=0.50, help="best 모델 선별 기준: 최소 accuracy")
args = parser.parse_args()

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


# ============================================================================
# 경로/유틸
# ============================================================================

def get_latest_light_model_run(models_base: Path):
    """7. light_models 안에서 checkpoints/best.keras 가 있는 run 중 이름 기준 최신 폴더 반환."""
    if not models_base.exists():
        return None
    candidates = []
    for d in models_base.iterdir():
        if not d.is_dir():
            continue
        ckpt = d / "checkpoints" / "best.keras"
        if ckpt.exists():
            candidates.append((d.name, d, ckpt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]  # (name, dir, best_ckpt_path)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_key(metrics: dict, target_recall: float, target_accuracy: float):
    """목표(recall/accuracy) 우선순위 기반 비교 키."""
    recall = _to_float(metrics.get("recall"))
    accuracy = _to_float(metrics.get("accuracy"))
    f1 = _to_float(metrics.get("f1"))
    meets_target = int(
        (recall is not None and accuracy is not None and recall >= target_recall and accuracy >= target_accuracy)
    )
    recall_gap = (recall - target_recall) if recall is not None else -1e9
    accuracy_gap = (accuracy - target_accuracy) if accuracy is not None else -1e9
    return (
        meets_target,
        recall_gap,
        accuracy_gap,
        recall if recall is not None else -1e9,
        accuracy if accuracy is not None else -1e9,
        f1 if f1 is not None else -1e9,
    )


def _extract_candidate_from_report(report_path: Path):
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    model_run = data.get("model_run")
    cls = data.get("classification", {})
    metrics = {
        "accuracy": _to_float(cls.get("accuracy")),
        "precision": _to_float(cls.get("precision")),
        "recall": _to_float(cls.get("recall")),
        "f1": _to_float(cls.get("f1_score")),
    }
    if not model_run or metrics["recall"] is None or metrics["f1"] is None:
        return None
    if metrics["accuracy"] is None:
        return None

    return {
        "model_run": model_run,
        "metrics": metrics,
        "report_path": str(report_path),
        "evaluation_time": data.get("evaluation_time"),
    }


def _collect_best_candidate_from_reports(eval_base_dir: Path, target_recall: float, target_accuracy: float):
    report_paths = sorted(eval_base_dir.glob("*/evaluation_report.json"))
    if not report_paths:
        return None

    best_by_run = {}
    for report_path in report_paths:
        candidate = _extract_candidate_from_report(report_path)
        if candidate is None:
            continue
        run_name = candidate["model_run"]
        prev = best_by_run.get(run_name)
        if prev is None:
            best_by_run[run_name] = candidate
            continue
        if _rank_key(candidate["metrics"], target_recall, target_accuracy) > _rank_key(prev["metrics"], target_recall, target_accuracy):
            best_by_run[run_name] = candidate

    if not best_by_run:
        return None

    return max(best_by_run.values(), key=lambda c: _rank_key(c["metrics"], target_recall, target_accuracy))


def _load_current_best_metrics(best_alias_dir: Path):
    """기존 light_model_best 성능을 읽어서 비교용 metrics 반환."""
    meta_path = best_alias_dir / "best_model_selection.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected = meta.get("selected", {})
            metrics = selected.get("metrics", {})
            return {
                "model_run": selected.get("model_run"),
                "metrics": {
                    "accuracy": _to_float(metrics.get("accuracy")),
                    "precision": _to_float(metrics.get("precision")),
                    "recall": _to_float(metrics.get("recall")),
                    "f1": _to_float(metrics.get("f1")),
                },
            }
        except Exception:
            pass

    # 메타가 없으면 light_model_best 내부 학습 평가 결과로 추론
    eval_result_path = best_alias_dir / "results" / "evaluation_results.json"
    if not eval_result_path.exists():
        return None
    try:
        with open(eval_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls = data.get("classification", {})
        accuracy = _to_float(cls.get("accuracy"))
        cm = cls.get("confusion_matrix")
        if accuracy is None and isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
            total = float(np.sum(cm))
            if total > 0:
                accuracy = (float(cm[0][0]) + float(cm[1][1])) / total
        return {
            "model_run": data.get("model_run"),
            "metrics": {
                "accuracy": accuracy,
                "precision": _to_float(cls.get("precision")),
                "recall": _to_float(cls.get("recall")),
                "f1": _to_float(cls.get("f1_score")),
            },
        }
    except Exception:
        return None





# ============================================================================
# 4. merge_data / 5. edit_data 로부터 테스트 데이터 구축
# ============================================================================

def _load_set_light_train_data_module():
    """6. set_light_train_data 모듈을 importlib로 로드 (파일명에 숫자 포함)."""
    script_path = Path(current_dir) / "6. set_light_train_data.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("set_light_train_data", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["set_light_train_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_test_data_from_merge_and_edit(
    merge_data_dir: str = "4. merge_data",
    edit_data_dir: str = "5. edit_data",
    min_points: int = 200,
    max_points: int = 400,
    sort_by: str = "height",
    max_normal_samples: int = 5000,
):
    """
    `4. merge_data/normal` 과 `5. edit_data/break` 에서 수집한 뒤
    (N, 304, 19, 3), (N,) 라벨 배열, 그리고 샘플 순서와 동일한 CSV 경로 리스트 반환.
    6. set_light_train_data 와 동일한 전처리 사용.
    """
    std = _load_set_light_train_data_module()
    if std is None:
        raise FileNotFoundError("6. set_light_train_data.py 를 찾을 수 없습니다.")

    prepare_sequence_from_csv = std.prepare_sequence_from_csv
    resize_img_height = std.resize_img_height
    collect_break_files_from_edit_data = std.collect_break_files_from_edit_data
    collect_all_crop_files = std.collect_all_crop_files

    # set_light_train_data 의 current_dir 은 make_ai 로 동일
    crop_files = collect_break_files_from_edit_data(
        edit_data_dir=edit_data_dir,
        merge_data_dir=merge_data_dir,
    )
    normal_files = collect_all_crop_files(merge_data_dir, is_break=False)

    imgs = []
    labels_cls = []
    csv_paths = []
    target_h = 304

    # 파단 데이터 전처리 진행 상황 표시
    from tqdm import tqdm

    print(f"  파단 CSV 개수: {len(crop_files)}개")
    for csv_path, _pn, _pid, label in tqdm(crop_files, desc="  [break] CSV 전처리", unit="file"):
        result = prepare_sequence_from_csv(csv_path=csv_path, sort_by=sort_by, feature_min_max=None)
        if result is None:
            continue
        img, meta = result
        n_pts = meta.get("original_length", 0)
        if n_pts < min_points or n_pts > max_points:
            continue
        img = resize_img_height(img, target_h=target_h)
        imgs.append(img)
        labels_cls.append(label)
        csv_paths.append(str(csv_path))

    n_kept_normal = 0
    print(f"  정상 CSV 개수: {len(normal_files)}개 (최대 {max_normal_samples}개 사용)")
    for csv_path, _pn, _pid, label in tqdm(normal_files, desc="  [normal] CSV 전처리", unit="file"):
        if n_kept_normal >= max_normal_samples:
            break
        result = prepare_sequence_from_csv(csv_path=csv_path, sort_by=sort_by, feature_min_max=None)
        if result is None:
            continue
        img, meta = result
        n_pts = meta.get("original_length", 0)
        if n_pts < min_points or n_pts > max_points:
            continue
        img = resize_img_height(img, target_h=target_h)
        imgs.append(img)
        labels_cls.append(label)
        csv_paths.append(str(csv_path))
        n_kept_normal += 1

    if not imgs:
        return None, None, None
    X = np.array(imgs, dtype=np.float32)
    y = np.array(labels_cls, dtype=np.int32)
    return X, y, csv_paths


# ============================================================================
# 1) 최신 모델 및 테스트 데이터 결정
# ============================================================================

models_base = Path(current_dir) / "7. light_models"
run_name_override = args.run
if run_name_override:
    print(f"지정 run으로 평가: {run_name_override}")

if run_name_override:
    run_dir = models_base / run_name_override
    best_ckpt_path = run_dir / "checkpoints" / "best.keras"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"지정 run 모델이 없습니다: {best_ckpt_path}")
    run_name = run_name_override
else:
    latest = get_latest_light_model_run(models_base)
    if latest is None:
        raise FileNotFoundError(
            f"7. light_models 아래에 checkpoints/best.keras 가 있는 run이 없습니다: {models_base}"
        )
    run_name, run_dir, best_ckpt_path = latest
    print(f"--run 미지정: 최신 run 자동 선택 -> {run_name}")
results_dir = run_dir / "results"
config_path = results_dir / "training_config.json"
# 평가 결과 저장: make_ai/8. evaluate_light_model/<날짜_시분>/
evaluate_light_model_dir = Path(current_dir) / "8. evaluate_light_model"
detail_dir = evaluate_light_model_dir / run_name
detail_dir.mkdir(parents=True, exist_ok=True)

merge_dir = Path(current_dir) / "4. merge_data"
edit_dir = Path(current_dir) / "5. edit_data"
if not merge_dir.exists():
    raise FileNotFoundError(f"4. merge_data 경로가 없습니다: {merge_dir}")
if not edit_dir.exists():
    raise FileNotFoundError(f"5. edit_data 경로가 없습니다: {edit_dir}")

print(f"\n사용 모델 run: {run_name}")
print(f"모델 경로: {best_ckpt_path}")
print(f"테스트 데이터 소스: 4. merge_data/normal, 5. edit_data/break")
print(f"상세 평가 결과 저장: {detail_dir}")


# ============================================================================
# 2) 테스트 데이터: 학습에 사용하지 않은 데이터 사용
#    - 우선 6. light_train_data/<학습 시 사용한 run>/test/ 사용 (학습 시 9:1로 분리된 10% 미사용 데이터)
#    - 없으면 4.+5. 수집 후 20% 분할로 테스트 구성
# ============================================================================

# 전체 데이터(4.+5.) 수집: "all" 이미지 및 validation(threshold)용
print("\n전체 데이터 수집 중 (4. merge_data/normal, 5. edit_data/break)...")
X_all, y_all, csv_paths_all = build_test_data_from_merge_and_edit(
    merge_data_dir="4. merge_data",
    edit_data_dir="5. edit_data",
    min_points=200,
    max_points=400,
    sort_by="height",
    max_normal_samples=5000,
)
if X_all is None or len(X_all) == 0:
    raise RuntimeError(
        "4. merge_data/normal 또는 5. edit_data/break 에서 유효한 샘플을 찾지 못했습니다. "
        "min_points=200, max_points=400 조건을 만족하는 CSV가 필요합니다."
    )

# 학습에 사용한 데이터 run 확인 (training_config.json의 data_source_run)
test_from_light_train_data = False
data_source_run = None
if config_path.exists():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)
        data_source_run = training_config.get("data_source_run")
    except Exception:
        pass

light_train_data_base = Path(current_dir) / "6. light_train_data"
if data_source_run and light_train_data_base.exists():
    test_imgs_path = light_train_data_base / data_source_run / "test" / "break_imgs_test.npy"
    test_lab_path = light_train_data_base / data_source_run / "test" / "break_labels_test.npy"
    if test_imgs_path.exists() and test_lab_path.exists():
        X_test = np.load(test_imgs_path).astype(np.float32)
        y_test_load = np.load(test_lab_path)
        y_test_flat = y_test_load.ravel().astype(np.int32)
        csv_paths_test = []
        test_from_light_train_data = True
        print(f"\n테스트 데이터: 6. light_train_data/{data_source_run}/test/ 사용 (학습에 사용하지 않은 10% 데이터)")
        n_total = len(X_test)
        n_normal = int((y_test_flat == 0).sum())
        n_break = int((y_test_flat == 1).sum())
        print(f"  테스트 샘플: {n_total} (normal={n_normal}, break={n_break})")
        y_test = y_test_flat.reshape(-1, 1).astype(np.float32)
        y_test_cls = y_test_flat
        # Validation(threshold용): 전체(4.+5.)에서 20% 분할
        indices = np.arange(len(X_all))
        _, idx_val = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)
        X_val = X_all[idx_val]
        y_val_flat = y_all[idx_val]
        print(f"  검증(threshold용, 4.+5. 20%): {len(X_val)}")
        test_data_source_desc = f"6. light_train_data/{data_source_run}/test (학습 미사용 10%)"

if not test_from_light_train_data:
    # 6. light_train_data test 없음: 4.+5. 수집 후 20% 테스트, 80% 검증으로 분할
    print("\n테스트 데이터: 6. light_train_data test 미사용. 4.+5. 수집 후 20% 분할로 테스트 구성.")
    indices = np.arange(len(X_all))
    idx_val, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_all
    )
    X_test = X_all[idx_test]
    X_val = X_all[idx_val]
    y_test_flat = y_all[idx_test]
    y_val_flat = y_all[idx_val]
    csv_paths_test = [csv_paths_all[i] for i in idx_test]
    y_test = y_test_flat.reshape(-1, 1).astype(np.float32)
    y_test_cls = y_test_flat.astype(int)
    n_total = len(X_test)
    n_normal = (y_test_cls == 0).sum()
    n_break = (y_test_cls == 1).sum()
    print(f"전체 수집: {len(X_all)} (normal={(y_all==0).sum()}, break={(y_all==1).sum()})")
    print(f"분할 후 테스트: {n_total} (normal={n_normal}, break={n_break})")
    print(f"분할 후 검증(threshold용): {len(X_val)}")
    test_data_source_desc = "4. merge_data + 5. edit_data (20% 분할, 6. light_train_data test 미사용)"

print(f"X_test: {X_test.shape}, dtype: {X_test.dtype}")


# ============================================================================
# 3) 모델 로드
# ============================================================================

print(f"\n모델 로드 중: {best_ckpt_path}")
model = keras.models.load_model(str(best_ckpt_path), compile=False)
print("모델 로드 완료.")
print("\n모델 구조 요약:")
model.summary()


# ============================================================================
# 4) 예측
# ============================================================================

BATCH = 32
print(f"\n예측 수행 중... (batch_size={BATCH})")
pred = model.predict(X_test, batch_size=BATCH, verbose=1)
y_prob = np.asarray(pred).reshape(-1)
print(f"y_prob shape: {y_prob.shape}, min/max: [{y_prob.min():.4f}, {y_prob.max():.4f}]")


# ============================================================================
# 5) Validation set에서 최적 threshold (4+5 데이터에서 분할한 20% 사용)
# ============================================================================

best_th = 0.5
val_used = False
if len(X_val) > 0 and len(np.unique(y_val_flat)) > 1:
    pred_val = model.predict(X_val, batch_size=BATCH, verbose=0)
    y_prob_val = np.asarray(pred_val).reshape(-1)
    y_val_cls = y_val_flat.astype(int)
    P_MIN = 0.50
    ths = np.linspace(0.01, 0.99, 199)
    best_tup = None
    for th in ths:
        yp = (y_prob_val >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val_cls, yp, average="binary", pos_label=1, zero_division=0
        )
        if p >= P_MIN and (best_tup is None or r > best_tup[2]):
            best_tup = (th, p, r, f1)
    if best_tup:
        best_th = best_tup[0]
        val_used = True
        print(f"Validation(4+5 데이터 20%) 기반 최적 threshold: {best_th:.4f} (Precision={best_tup[1]:.4f}, Recall={best_tup[2]:.4f}, F1={best_tup[3]:.4f})")
if not val_used:
    print("Validation 미사용. threshold=0.5 으로 평가.")

y_pred = (y_prob >= best_th).astype(int)


# ============================================================================
# 6) 분류 지표
# ============================================================================

precision = precision_score(y_test_cls, y_pred, zero_division=0)
recall = recall_score(y_test_cls, y_pred, zero_division=0)
f1 = f1_score(y_test_cls, y_pred, zero_division=0)
acc = accuracy_score(y_test_cls, y_pred)
roc_auc = roc_auc_score(y_test_cls, y_prob)
fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
prec_curve, rec_curve, _ = precision_recall_curve(y_test_cls, y_prob)
pr_auc = sk_metrics.auc(rec_curve, prec_curve)

cm = confusion_matrix(y_test_cls, y_pred)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

print("\n" + "=" * 60)
print("Classification 평가 (Test)")
print("=" * 60)
print(f"Threshold: {best_th:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred, digits=4, zero_division=0, target_names=["Normal(0)", "Break(1)"]))


# ============================================================================
# 6-1) Threshold 구간별 FN/Recall/Precision (파단 놓침 감소 가능성 분석)
# ============================================================================

n_break_true = int((y_test_cls == 1).sum())
ths_sens = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
threshold_sensitivity = []
for th in ths_sens:
    yp = (y_prob >= th).astype(int)
    fn = int(((y_test_cls == 1) & (yp == 0)).sum())
    fp = int(((y_test_cls == 0) & (yp == 1)).sum())
    r = recall_score(y_test_cls, yp, zero_division=0)
    p = precision_score(y_test_cls, yp, zero_division=0)
    fn_ratio = fn / n_break_true if n_break_true else 0
    threshold_sensitivity.append({
        "threshold": round(th, 2),
        "fn_count": fn,
        "fn_ratio_pct": round(100 * fn_ratio, 2),
        "recall": round(r, 4),
        "precision": round(p, 4),
        "fp_count": fp,
    })

print("\n[파단 놓침(FN) 감소 가능성] Threshold 구간별 (Test set 기준)")
print("  (파단인데 정상으로 판단한 비율을 줄이려면 threshold 낮추기 검토)")
print("-" * 72)
print(f"  {'th':>5}  {'FN':>4}  {'FN%':>6}  {'Recall':>7}  {'Precision':>10}  {'FP':>5}")
print("-" * 72)
for row in threshold_sensitivity:
    print(f"  {row['threshold']:>5.2f}  {row['fn_count']:>4}  {row['fn_ratio_pct']:>5.1f}%  {row['recall']:>7.4f}  {row['precision']:>10.4f}  {row['fp_count']:>5}")
print("-" * 72)

th_sens_path = detail_dir / "threshold_sensitivity.json"
with open(th_sens_path, "w", encoding="utf-8") as f:
    json.dump({"n_break_true": n_break_true, "rows": threshold_sensitivity}, f, ensure_ascii=False, indent=2)
print(f"저장: {th_sens_path}")


# ============================================================================
# 7) 시각화 1: 혼동행렬
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cm, cmap="Blues")
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Normal (0)", "Break (1)"])
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Normal (0)", "Break (1)"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")

axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Normal (0)", "Break (1)"])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Normal (0)", "Break (1)"])
for i in range(2):
    for j in range(2):
        pct = cm_norm[i, j] * 100
        axes[1].text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center", fontsize=11)
plt.tight_layout()
cm_path = detail_dir / "test_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {cm_path}")


# ============================================================================
# 8) 시각화 2: ROC & PR Curve
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})", lw=2)
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(rec_curve, prec_curve, label=f"PR (AUC={pr_auc:.4f})", lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
roc_path = detail_dir / "test_roc_pr_curves.png"
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {roc_path}")


# ============================================================================
# 9) 시각화 3: Threshold sweep (Precision / Recall / F1 vs threshold)
# ============================================================================

ths_plot = np.linspace(0.01, 0.99, 99)
precs, recs, f1s = [], [], []
for t in ths_plot:
    yp = (y_prob >= t).astype(int)
    p = precision_score(y_test_cls, yp, zero_division=0)
    r = recall_score(y_test_cls, yp, zero_division=0)
    f = f1_score(y_test_cls, yp, zero_division=0)
    precs.append(p)
    recs.append(r)
    f1s.append(f)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ths_plot, precs, label="Precision", lw=1.5)
ax.plot(ths_plot, recs, label="Recall", lw=1.5)
ax.plot(ths_plot, f1s, label="F1-Score", lw=1.5)
ax.axvline(best_th, color="gray", ls="--", label=f"Used th={best_th:.3f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs Threshold (Test Set)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
th_path = detail_dir / "test_threshold_sweep.png"
plt.savefig(th_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {th_path}")


# ============================================================================
# 10) 시각화 4: 예측 확률 분포 (클래스별 / 오분류 각각 이미지)
# ============================================================================
mask_n = y_test_cls == 0
mask_b = y_test_cls == 1
fp = (y_test_cls == 0) & (y_pred == 1)
fn = (y_test_cls == 1) & (y_pred == 0)

# 4-1) 클래스별 예측 확률 분포 (Normal / Break)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_prob[mask_n], bins=30, alpha=0.7, label=f"Normal (n={mask_n.sum()})", color="C0", density=True)
ax.hist(y_prob[mask_b], bins=30, alpha=0.7, label=f"Break (n={mask_b.sum()})", color="C1", density=True)
ax.axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
ax.set_xlabel("Predicted P(break)")
ax.set_ylabel("Density")
ax.set_title("Prediction Score Distribution by True Class (Test)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
dist_by_class_path = detail_dir / "test_prediction_distribution_by_class.png"
plt.savefig(dist_by_class_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {dist_by_class_path}")

# 4-2) 오분류만: FP, FN (Test 기준)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_prob[fp], bins=20, alpha=0.7, label=f"False Positive (n={fp.sum()})", color="orange")
ax.hist(y_prob[fn], bins=20, alpha=0.7, label=f"False Negative (n={fn.sum()})", color="red")
ax.axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
ax.set_xlabel("Predicted P(break)")
ax.set_ylabel("Count")
ax.set_title("Misclassified Samples: Score Distribution (Test)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
dist_misclassified_path = detail_dir / "test_prediction_distribution_misclassified.png"
plt.savefig(dist_misclassified_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {dist_misclassified_path}")

# ============================================================================
# 10-2) 시각화: TP, TN 예측 확률 분포 (정분류, FP/FN과 같은 방식)
# ============================================================================
tp_mask = (y_test_cls == 1) & (y_pred == 1)
tn_mask = (y_test_cls == 0) & (y_pred == 0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_prob[tp_mask], bins=20, alpha=0.7, label=f"True Positive (n={tp_mask.sum()})", color="green")
ax.hist(y_prob[tn_mask], bins=20, alpha=0.7, label=f"True Negative (n={tn_mask.sum()})", color="blue")
ax.axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
ax.set_xlabel("Predicted P(break)")
ax.set_ylabel("Count")
ax.set_title("Correctly Classified Samples: TP / TN Score Distribution (Test)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
tp_tn_dist_path = detail_dir / "test_tp_tn_distribution.png"
plt.savefig(tp_tn_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {tp_tn_dist_path}")


# ============================================================================
# 10-1) 전체 데이터(X_all) 기준 성능 및 프로젝트 개요 이미지
# ============================================================================

print("\n전체 데이터 기준 성능 계산 중 (Validation에서 선택한 threshold 사용)...")
y_prob_all = model.predict(X_all, batch_size=BATCH, verbose=0).reshape(-1)
y_pred_all = (y_prob_all >= best_th).astype(int)
y_all_cls = y_all.astype(int)

acc_all = accuracy_score(y_all_cls, y_pred_all)
precision_all = precision_score(y_all_cls, y_pred_all, zero_division=0)
recall_all = recall_score(y_all_cls, y_pred_all, zero_division=0)
f1_all = f1_score(y_all_cls, y_pred_all, zero_division=0)
roc_auc_all = roc_auc_score(y_all_cls, y_prob_all)
fpr_all, tpr_all, _ = roc_curve(y_all_cls, y_prob_all)
prec_curve_all, rec_curve_all, _ = precision_recall_curve(y_all_cls, y_prob_all)
pr_auc_all = sk_metrics.auc(rec_curve_all, prec_curve_all)
cm_all = confusion_matrix(y_all_cls, y_pred_all)
cm_all_norm = cm_all / (cm_all.sum(axis=1, keepdims=True) + 1e-12)

print("전체 데이터 Accuracy/Precision/Recall/F1:", acc_all, precision_all, recall_all, f1_all)

# ============================================================================
# All 데이터 기준 시각화 (test와 동일 구성, 전체 데이터로 계산)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cm_all, cmap="Blues")
axes[0].set_title("Confusion Matrix (Counts) [All Data]")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Normal (0)", "Break (1)"])
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Normal (0)", "Break (1)"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm_all[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
axes[1].imshow(cm_all_norm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix (Normalized) [All Data]")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Normal (0)", "Break (1)"])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Normal (0)", "Break (1)"])
for i in range(2):
    for j in range(2):
        pct = cm_all_norm[i, j] * 100
        axes[1].text(j, i, f"{cm_all[i, j]}\n({pct:.1f}%)", ha="center", va="center", fontsize=11)
plt.tight_layout()
cm_all_path = detail_dir / "all_confusion_matrix.png"
plt.savefig(cm_all_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {cm_all_path}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr_all, tpr_all, label=f"ROC (AUC={roc_auc_all:.4f})", lw=2)
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve [All Data]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(rec_curve_all, prec_curve_all, label=f"PR (AUC={pr_auc_all:.4f})", lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve [All Data]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
roc_all_path = detail_dir / "all_roc_pr_curves.png"
plt.savefig(roc_all_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {roc_all_path}")

ths_plot_all = np.linspace(0.01, 0.99, 99)
precs_all, recs_all, f1s_all = [], [], []
for t in ths_plot_all:
    yp = (y_prob_all >= t).astype(int)
    p = precision_score(y_all_cls, yp, zero_division=0)
    r = recall_score(y_all_cls, yp, zero_division=0)
    f = f1_score(y_all_cls, yp, zero_division=0)
    precs_all.append(p)
    recs_all.append(r)
    f1s_all.append(f)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ths_plot_all, precs_all, label="Precision", lw=1.5)
ax.plot(ths_plot_all, recs_all, label="Recall", lw=1.5)
ax.plot(ths_plot_all, f1s_all, label="F1-Score", lw=1.5)
ax.axvline(best_th, color="gray", ls="--", label=f"Used th={best_th:.3f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs Threshold [All Data]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
th_all_path = detail_dir / "all_threshold_sweep.png"
plt.savefig(th_all_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {th_all_path}")

fp_all = (y_all_cls == 0) & (y_pred_all == 1)
fn_all = (y_all_cls == 1) & (y_pred_all == 0)
tp_all = (y_all_cls == 1) & (y_pred_all == 1)
tn_all = (y_all_cls == 0) & (y_pred_all == 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mask_n_all = y_all_cls == 0
mask_b_all = y_all_cls == 1
axes[0].hist(y_prob_all[mask_n_all], bins=30, alpha=0.7, label=f"Normal (n={mask_n_all.sum()})", color="C0", density=True)
axes[0].hist(y_prob_all[mask_b_all], bins=30, alpha=0.7, label=f"Break (n={mask_b_all.sum()})", color="C1", density=True)
axes[0].axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
axes[0].set_xlabel("Predicted P(break)")
axes[0].set_ylabel("Density")
axes[0].set_title("Prediction Score Distribution by True Class [All Data]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].hist(y_prob_all[fp_all], bins=20, alpha=0.7, label=f"False Positive (n={fp_all.sum()})", color="orange")
axes[1].hist(y_prob_all[fn_all], bins=20, alpha=0.7, label=f"False Negative (n={fn_all.sum()})", color="red")
axes[1].axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
axes[1].set_xlabel("Predicted P(break)")
axes[1].set_ylabel("Count")
axes[1].set_title("Misclassified Samples: Score Distribution [All Data]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
dist_all_path = detail_dir / "all_prediction_distribution.png"
plt.savefig(dist_all_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {dist_all_path}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_prob_all[tp_all], bins=20, alpha=0.7, label=f"True Positive (n={tp_all.sum()})", color="green")
ax.hist(y_prob_all[tn_all], bins=20, alpha=0.7, label=f"True Negative (n={tn_all.sum()})", color="blue")
ax.axvline(best_th, color="black", ls="--", label=f"threshold={best_th:.3f}")
ax.set_xlabel("Predicted P(break)")
ax.set_ylabel("Count")
ax.set_title("Correctly Classified: TP / TN Score Distribution [All Data]")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
tp_tn_all_path = detail_dir / "all_tp_tn_distribution.png"
plt.savefig(tp_tn_all_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {tp_tn_all_path}")

# ============================================================================
# 10-1) 전체 데이터(X_all) 기준 성능 및 프로젝트 개요 이미지
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (1) Test 기준 정규화 혼동행렬 – 파단 감지 성능
im = axes[0, 0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
axes[0, 0].set_title("Test Confusion Matrix (Normalized)\nPole Break Detection")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("True")
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(["Normal", "Break"])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_yticklabels(["Normal", "Break"])
for i in range(2):
    for j in range(2):
        pct = cm_norm[i, j] * 100
        axes[0, 0].text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

# (2) Test 기준 Precision / Recall / F1 막대 그래프
metrics_names = ["Precision", "Recall", "F1-Score"]
metrics_vals = [precision, recall, f1]
axes[0, 1].bar(metrics_names, metrics_vals, color=["C0", "C1", "C2"])
for x, v in zip(metrics_names, metrics_vals):
    axes[0, 1].text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].set_ylabel("Score")
axes[0, 1].set_title("Test Set: Classification Metrics")
axes[0, 1].grid(True, axis="y", alpha=0.3)

# (3) 전체 데이터 클래스 분포 – 실제 현장 데이터 구성
total_normal = int((y_all_cls == 0).sum())
total_break = int((y_all_cls == 1).sum())
axes[1, 0].bar(["Normal", "Break"], [total_normal, total_break], color=["C0", "C1"])
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Class Distribution (All Data)\n(4. merge_data/normal + 5. edit_data/break)")
for x, v in zip(["Normal", "Break"], [total_normal, total_break]):
    axes[1, 0].text(x, v, f"{v}", ha="center", va="bottom")
axes[1, 0].grid(True, axis="y", alpha=0.3)

# (4) 텍스트 요약 – 전체 데이터 vs Test 기준 성능 비교
axes[1, 1].axis("off")
fn_count_test = int(fn.sum())
fp_count_test = int(fp.sum())
fn_rate_test = (fn_count_test / n_break * 100.0) if n_break else 0.0
overview_lines = [
    "SMARTCS Pole - Pole Break Detection Summary",
    "",
    "[Data Source] 4. merge_data/normal, 5. edit_data/break",
    f"  Total samples: {len(X_all)} (Normal={total_normal}, Break={total_break})",
    f"  Test samples: {n_total} (Normal={n_normal}, Break={n_break})",
    "",
    "[Test Set]",
    f"  Accuracy : {acc:.3f}",
    f"  Precision: {precision:.3f}",
    f"  Recall   : {recall:.3f}",
    f"  F1-Score : {f1:.3f}",
    f"  ROC-AUC  : {roc_auc:.3f}, PR-AUC: {pr_auc:.3f}",
    f"  FN (missed break): {fn_count_test} ({fn_rate_test:.2f}%)",
    "",
    "[All Data]",
    f"  Accuracy : {acc_all:.3f}",
    f"  Precision: {precision_all:.3f}",
    f"  Recall   : {recall_all:.3f}",
    f"  F1-Score : {f1_all:.3f}",
    f"  ROC-AUC  : {roc_auc_all:.3f}, PR-AUC: {pr_auc_all:.3f}",
]
axes[1, 1].text(
    0.0,
    1.0,
    "\n".join(overview_lines),
    ha="left",
    va="top",
    fontsize=9,
)

plt.tight_layout()
overview_path = detail_dir / "overview_summary.png"
plt.savefig(overview_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {overview_path}")


# ============================================================================
# 11) 오분류 요약 (인덱스·통계)
# ============================================================================

fp_indices = np.where(fp)[0].tolist()
fn_indices = np.where(fn)[0].tolist()
tp_indices = np.where((y_test_cls == 1) & (y_pred == 1))[0].tolist()
tn_indices = np.where((y_test_cls == 0) & (y_pred == 0))[0].tolist()

misclass_summary = {
    "false_positive": {"count": len(fp_indices), "indices_sample": fp_indices[:50]},
    "false_negative": {"count": len(fn_indices), "indices_sample": fn_indices[:50]},
    "true_positive": {"count": len(tp_indices), "indices_sample": tp_indices[:50]},
    "true_negative": {"count": len(tn_indices), "indices_sample": tn_indices[:50]},
}
if fp_indices:
    misclass_summary["false_positive"]["prob_mean"] = float(y_prob[fp].mean())
    misclass_summary["false_positive"]["prob_std"] = float(y_prob[fp].std())
if fn_indices:
    misclass_summary["false_negative"]["prob_mean"] = float(y_prob[fn].mean())
    misclass_summary["false_negative"]["prob_std"] = float(y_prob[fn].std())
if tp_indices:
    misclass_summary["true_positive"]["prob_mean"] = float(y_prob[tp_mask].mean())
    misclass_summary["true_positive"]["prob_std"] = float(y_prob[tp_mask].std())
if tn_indices:
    misclass_summary["true_negative"]["prob_mean"] = float(y_prob[tn_mask].mean())
    misclass_summary["true_negative"]["prob_std"] = float(y_prob[tn_mask].std())


# ============================================================================
# 12) 상세 평가 결과 JSON 저장
# ============================================================================

eval_report = {
    "model_run": run_name,
    "model_path": str(best_ckpt_path),
    "test_data_source": test_data_source_desc,
    "all_data_source": "4. merge_data/normal, 5. edit_data/break",
    "evaluation_time": datetime.datetime.now().isoformat(),
    "test_samples": n_total,
    "class_distribution": {"normal": int(n_normal), "break": int(n_break)},
    "threshold": float(best_th),
    "threshold_source": "validation" if val_used else "default_0.5",
    "classification": {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": [[float(x) for x in row] for row in cm_norm.tolist()],
        "all_data_metrics": {
            "accuracy": float(acc_all),
            "precision": float(precision_all),
            "recall": float(recall_all),
            "f1_score": float(f1_all),
            "roc_auc": float(roc_auc_all),
            "pr_auc": float(pr_auc_all),
            "confusion_matrix": cm_all.tolist(),
            "confusion_matrix_normalized": [[float(x) for x in row] for row in cm_all_norm.tolist()],
        },
    },
    "misclassification": misclass_summary,
    "outputs": {
        "test_confusion_matrix": str(cm_path),
        "test_roc_pr_curves": str(roc_path),
        "test_threshold_sweep": str(th_path),
        "test_prediction_distribution_by_class": str(dist_by_class_path),
        "test_prediction_distribution_misclassified": str(dist_misclassified_path),
        "test_tp_tn_distribution": str(tp_tn_dist_path),
        "all_confusion_matrix": str(cm_all_path),
        "all_roc_pr_curves": str(roc_all_path),
        "all_threshold_sweep": str(th_all_path),
        "all_prediction_distribution": str(dist_all_path),
        "all_tp_tn_distribution": str(tp_tn_all_path),
        "overview_summary": str(overview_path),
        "threshold_sensitivity": str(th_sens_path),
    },
    "threshold_sensitivity": {"n_break_true": n_break_true, "rows": threshold_sensitivity},
}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    eval_report["training_config_summary"] = {
        "data_source_run": config.get("data_source_run"),
        "total_epochs_trained": config.get("total_epochs_trained"),
        "loss": config.get("loss"),
        "input_shape": config.get("data", {}).get("input_shape") or config.get("model", {}).get("input_shape"),
    }

report_path = detail_dir / "evaluation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, ensure_ascii=False, indent=2)
print(f"저장: {report_path}")

if args.target_pass_mode == "recall_priority":
    overall_pass = (recall >= args.target_recall) and (f1 >= args.target_f1)
else:
    overall_pass = (
        (precision >= args.target_precision)
        and (recall >= args.target_recall)
        and (f1 >= args.target_f1)
    )

feedback = {
    "stage": "light_model",
    "evaluation_dir": str(detail_dir),
    "model_run": run_name,
    "pass": bool(overall_pass),
    "recommended_retrain": bool(not overall_pass),
    "pass_mode": args.target_pass_mode,
    "criteria": {
        "target_precision": float(args.target_precision),
        "target_recall": float(args.target_recall),
        "target_f1": float(args.target_f1),
    },
    "actual": {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "threshold": float(best_th),
        "fn_count": int(len(fn_indices)),
    },
}
feedback_path = detail_dir / "training_feedback.json"
with open(feedback_path, "w", encoding="utf-8") as f:
    json.dump(feedback, f, ensure_ascii=False, indent=2)
print(f"저장: {feedback_path}")
print(
    "[feedback] pass={} mode={} precision={:.4f} recall={:.4f} f1={:.4f}".format(
        overall_pass, args.target_pass_mode, precision, recall, f1
    )
)

# ============================================================================
# 12-0) 누적 평가 보고서 기준 best 모델 선별 및 light_model_best 갱신
# ============================================================================

eval_base_dir = Path(current_dir) / "8. evaluate_light_model"
best_alias_dir = Path(current_dir) / "light_model_best"
best_candidate = _collect_best_candidate_from_reports(
    eval_base_dir=eval_base_dir,
    target_recall=float(args.best_target_recall),
    target_accuracy=float(args.best_target_accuracy),
)

if best_candidate is None:
    print("경고: 누적 evaluation_report.json에서 후보 모델을 찾지 못했습니다. light_model_best 갱신을 건너뜁니다.")
else:
    current_best = _load_current_best_metrics(best_alias_dir)
    should_replace = current_best is None
    if current_best is not None:
        should_replace = (
            _rank_key(best_candidate["metrics"], args.best_target_recall, args.best_target_accuracy)
            > _rank_key(current_best["metrics"], args.best_target_recall, args.best_target_accuracy)
        )

    selected_run_dir = models_base / best_candidate["model_run"]
    if not selected_run_dir.exists():
        print(f"경고: 선별된 모델 run 디렉터리가 없습니다: {selected_run_dir}")
    elif should_replace:
        if best_alias_dir.exists():
            shutil.rmtree(best_alias_dir)
        shutil.copytree(selected_run_dir, best_alias_dir)
        selection_meta = {
            "updated_at": datetime.datetime.now().isoformat(),
            "criteria": {
                "target_recall": float(args.best_target_recall),
                "target_accuracy": float(args.best_target_accuracy),
            },
            "selected": {
                "model_run": best_candidate["model_run"],
                "metrics": best_candidate["metrics"],
                "report_path": best_candidate["report_path"],
                "evaluation_time": best_candidate.get("evaluation_time"),
            },
            "previous": current_best,
        }
        with open(best_alias_dir / "best_model_selection.json", "w", encoding="utf-8") as f:
            json.dump(selection_meta, f, ensure_ascii=False, indent=2)
        print(
            "light_model_best 갱신: run={} (recall={:.4f}, accuracy={:.4f}, f1={:.4f})".format(
                best_candidate["model_run"],
                best_candidate["metrics"]["recall"],
                best_candidate["metrics"]["accuracy"],
                best_candidate["metrics"]["f1"],
            )
        )
    else:
        print(
            "light_model_best 유지: 기존 베스트가 더 우수하거나 동일함 "
            f"(current={current_best.get('model_run') if current_best else None}, "
            f"candidate={best_candidate['model_run']})"
        )


# ============================================================================
# 12-1) 파단 놓침(FN) 감소 방안 요약
# ============================================================================

fn_analysis_lines = [
    "=" * 70,
    "파단인데 정상으로 판단한 비율(FN) 줄이기 방안",
    "=" * 70,
    "",
    "[현재 상태]",
    f"  파단 실제 개수: {n_break_true}, FN(놓침): {len(fn_indices)}개, FN 비율: {100 * len(fn_indices) / n_break_true:.2f}%" if n_break_true else "  (파단 샘플 없음)",
    f"  사용 threshold: {best_th:.4f}",
    "",
    "[Threshold 구간별 FN/Recall/Precision] (threshold_sensitivity.json 참고)",
]
for row in threshold_sensitivity:
    fn_analysis_lines.append(
        f"  th={row['threshold']:.2f} → FN={row['fn_count']} ({row['fn_ratio_pct']}%), Recall={row['recall']:.4f}, Precision={row['precision']:.4f}, FP={row['fp_count']}"
    )
fn_analysis_lines.extend([
    "",
    "[FN을 더 줄이는 방법]",
    "  1. Threshold 낮추기: 현재보다 낮은 threshold(예: 0.40, 0.35) 사용 시 FN 감소, 대신 FP(정상인데 파단 판정) 증가.",
    "  2. 평가 스크립트의 최적 threshold 조건 변경: 8. evaluate_light_model.py 에서 P_MIN=0.50 을 더 낮추면 'Recall 우선' threshold가 선택됨.",
    "  3. 학습 시 Recall 중시: 7. make_light_model.py 에서 class_weight·Focal alpha를 파단 쪽으로 더 두거나, recall 기반 early-stop/체크포인트 고려.",
    "  4. FN 샘플 분석: 놓친 파단 패턴을 보고 데이터·증강 보완.",
    "=" * 70,
])
fn_analysis_path = detail_dir / "fn_reduction_analysis.txt"
with open(fn_analysis_path, "w", encoding="utf-8") as f:
    f.write("\n".join(fn_analysis_lines))
print(f"저장: {fn_analysis_path}")


# ============================================================================
# 13) 요약 텍스트 리포트
# ============================================================================

summary_lines = [
    "=" * 70,
    "Light 모델 상세 평가 요약",
    "=" * 70,
    f"모델 run: {run_name}",
    f"모델 경로: {best_ckpt_path}",
    "테스트 데이터 소스: 4. merge_data/normal, 5. edit_data/break",
    f"평가 시각: {eval_report['evaluation_time']}",
    "",
    f"테스트 샘플: {n_total} (Normal={n_normal}, Break={n_break})",
    f"사용 threshold: {best_th:.4f} ({eval_report['threshold_source']})",
    "",
    "[분류 성능]",
    f"  Accuracy:  {acc:.4f}",
    f"  Precision: {precision:.4f}",
    f"  Recall:    {recall:.4f}",
    f"  F1-Score:  {f1:.4f}",
    f"  ROC-AUC:   {roc_auc:.4f}",
    f"  PR-AUC:    {pr_auc:.4f}",
    "",
    "[혼동행렬]",
    f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
    "",
    "[오분류]",
    f"  False Positive: {len(fp_indices)}개",
    f"  False Negative: {len(fn_indices)}개",
    "",
    "[저장 파일]",
    f"  {detail_dir}",
    f"    - test_confusion_matrix.png",
    f"    - test_roc_pr_curves.png",
    f"    - test_threshold_sweep.png",
    f"    - test_prediction_distribution_by_class.png",
    f"    - test_prediction_distribution_misclassified.png",
    f"    - test_tp_tn_distribution.png",
    f"    - all_confusion_matrix.png",
    f"    - all_roc_pr_curves.png",
    f"    - all_threshold_sweep.png",
    f"    - all_prediction_distribution.png",
    f"    - all_tp_tn_distribution.png",
    f"    - overview_summary.png",
    f"    - evaluation_report.json",
    f"    - evaluation_summary.txt",
    f"    - image_descriptions.txt",
    f"    - threshold_sensitivity.json",
    f"    - fn_reduction_analysis.txt",
]
summary_lines.append("=" * 70)
summary_path = detail_dir / "evaluation_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"저장: {summary_path}")

# ============================================================================
# image_descriptions.txt: 각 이미지에 대한 설명
# ============================================================================
image_descriptions_lines = [
    "Light 모델(전주 파단 분류) 평가 이미지 설명:",
    "",
    "[테스트 데이터] 학습에 사용하지 않은 테스트 세트만으로 평가.",
    "test_confusion_matrix.png: 테스트 세트 기준 혼동행렬(개수 및 정규화).",
    "test_roc_pr_curves.png: 테스트 세트 기준 ROC 곡선 및 Precision-Recall 곡선.",
    "test_threshold_sweep.png: 테스트 세트 기준 threshold에 따른 Precision, Recall, F1-Score 변화.",
    "test_prediction_distribution_by_class.png: 테스트 세트 기준 실제 클래스(정상/파단)별 예측 점수 분포.",
    "test_prediction_distribution_misclassified.png: 테스트 세트 기준 오분류(FP/FN) 샘플의 예측 점수 분포.",
    "test_tp_tn_distribution.png: 테스트 세트 기준 정분류(TP/TN) 샘플의 예측 점수 분포.",
    "",
    "[전체 데이터] 4. merge_data + 5. edit_data 전체로 평가.",
    "all_confusion_matrix.png: 전체 데이터 기준 혼동행렬(개수 및 정규화).",
    "all_roc_pr_curves.png: 전체 데이터 기준 ROC 곡선 및 Precision-Recall 곡선.",
    "all_threshold_sweep.png: 전체 데이터 기준 threshold에 따른 Precision, Recall, F1-Score 변화.",
    "all_prediction_distribution.png: 전체 데이터 기준 클래스별·오분류(FP/FN) 예측 점수 분포.",
    "all_tp_tn_distribution.png: 전체 데이터 기준 정분류(TP/TN) 예측 점수 분포.",
    "",
    "[요약]",
    "overview_summary.png: 테스트 vs 전체 지표, 클래스 분포, 텍스트 비교를 담은 요약 그림.",
]
image_descriptions_path = detail_dir / "image_descriptions.txt"
with open(image_descriptions_path, "w", encoding="utf-8") as f:
    f.write("\n".join(image_descriptions_lines))
print(f"저장: {image_descriptions_path}")

print("\n" + "\n".join(summary_lines))
