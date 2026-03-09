#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _metric_delta(current: float | None, previous: float | None) -> dict[str, Any]:
    if current is None or previous is None:
        return {"current": current, "previous": previous, "delta": None}
    return {"current": current, "previous": previous, "delta": current - previous}


def _threshold_sweep_rows(y_true: np.ndarray, score: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.unique(np.round(score.astype(float), 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return rows


def _get_latest_train_data_run() -> Path | None:
    base = PROJECT_ROOT / "main" / "4. make_mlp_model" / "1. mlp_train_data"
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "X_val.npy").exists() and (d / "y_val.npy").exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]


def _adapt_features_for_model(x_val: np.ndarray, pipeline: Any) -> np.ndarray:
    expected = getattr(pipeline, "n_features_in_", None)
    if expected is None and hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            expected = getattr(step, "n_features_in_", None)
            if expected is not None:
                break
    if expected is None or x_val.shape[1] == expected:
        return x_val
    if x_val.shape[1] == 13 and expected == 7:
        return x_val[:, [0, 4, 5, 6, 8, 9, 11]]
    raise ValueError(f"지원하지 않는 feature shape 변환입니다: input={x_val.shape[1]}, expected={expected}")


def _load_package_metrics(package_dir: Path) -> dict[str, Any]:
    light = _load_json(package_dir / "light_model" / "test_results" / "light_test_metrics.json")
    hard2 = _load_json(package_dir / "hard_model_2nd" / "test_results" / "hard2_axis_test_metrics.json")
    mlp = _load_json(package_dir / "mlp_model" / "test_results" / "mlp_summary.json")
    if "overall_metrics" not in hard2:
        axis_metrics = hard2.get("axis_metrics", {})
        if axis_metrics:
            hard2["overall_metrics"] = {
                "best_f1": float(np.mean([axis_metrics[a]["metrics"].get("best_f1", 0.0) for a in axis_metrics])),
                "auc_high_iou": float(np.mean([axis_metrics[a]["metrics"].get("auc_high_iou", 0.0) for a in axis_metrics])),
                "separation": float(np.mean([axis_metrics[a]["metrics"].get("separation", 0.0) for a in axis_metrics])),
            }
    return {"light": light, "hard2": hard2, "mlp": mlp}


def _build_comparison(current: dict[str, Any], reference: dict[str, Any], reference_name: str) -> dict[str, Any]:
    axis_comp = {}
    for axis in ("x", "y", "z"):
        cur_axis = current["hard2"].get("axis_metrics", {}).get(axis, {}).get("metrics", {})
        ref_axis = reference["hard2"].get("axis_metrics", {}).get(axis, {}).get("metrics", {})
        axis_comp[axis] = {
            "best_f1": _metric_delta(_safe_float(cur_axis.get("best_f1")), _safe_float(ref_axis.get("best_f1"))),
            "auc_high_iou": _metric_delta(_safe_float(cur_axis.get("auc_high_iou")), _safe_float(ref_axis.get("auc_high_iou"))),
            "separation": _metric_delta(_safe_float(cur_axis.get("separation")), _safe_float(ref_axis.get("separation"))),
        }
    return {
        "compare_to": reference_name,
        "light": {
            "f1_score": _metric_delta(_safe_float(current["light"].get("f1_score")), _safe_float(reference["light"].get("f1_score"))),
            "recall": _metric_delta(_safe_float(current["light"].get("recall")), _safe_float(reference["light"].get("recall"))),
            "precision": _metric_delta(_safe_float(current["light"].get("precision")), _safe_float(reference["light"].get("precision"))),
            "roc_auc": _metric_delta(_safe_float(current["light"].get("roc_auc")), _safe_float(reference["light"].get("roc_auc"))),
        },
        "hard2": {
            "overall_best_f1": _metric_delta(_safe_float(current["hard2"]["overall_metrics"].get("best_f1")), _safe_float(reference["hard2"]["overall_metrics"].get("best_f1"))),
            "overall_auc_high_iou": _metric_delta(_safe_float(current["hard2"]["overall_metrics"].get("auc_high_iou")), _safe_float(reference["hard2"]["overall_metrics"].get("auc_high_iou"))),
            "overall_separation": _metric_delta(_safe_float(current["hard2"]["overall_metrics"].get("separation")), _safe_float(reference["hard2"]["overall_metrics"].get("separation"))),
            "by_axis": axis_comp,
        },
        "mlp": {
            "binary_alert_f1": _metric_delta(_safe_float(current["mlp"]["metrics"]["binary_alert"].get("f1")), _safe_float(reference["mlp"]["metrics"]["binary_alert"].get("f1"))),
            "binary_break_f1": _metric_delta(_safe_float(current["mlp"]["metrics"]["binary_break"].get("f1")), _safe_float(reference["mlp"]["metrics"]["binary_break"].get("f1"))),
            "binary_break_auc": _metric_delta(_safe_float(current["mlp"]["metrics"]["binary_break"].get("roc_auc")), _safe_float(reference["mlp"]["metrics"]["binary_break"].get("roc_auc"))),
        },
    }


def _save_bar_chart(labels: list[str], current_vals: list[float], ref_vals: list[float], title: str, out_path: Path) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ref_vals, width, color="#9E9E9E", label="Reference")
    ax.bar(x + width / 2, current_vals, width, color="#4C78A8", label="Current")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _generate_visuals(package_dir: Path, current: dict[str, Any], reference: dict[str, Any], reference_name: str, train_data_run: str | None = None) -> list[str]:
    visuals_dir = package_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    output: list[str] = []

    p = visuals_dir / "light_metric_comparison.png"
    _save_bar_chart(
        ["F1", "Recall", "Precision", "ROC AUC"],
        [float(current["light"]["f1_score"]), float(current["light"]["recall"]), float(current["light"]["precision"]), float(current["light"]["roc_auc"])],
        [float(reference["light"]["f1_score"]), float(reference["light"]["recall"]), float(reference["light"]["precision"]), float(reference["light"]["roc_auc"])],
        f"Light 비교 ({package_dir.name} vs {reference_name})",
        p,
    )
    output.append(str(p.resolve()))

    p = visuals_dir / "mlp_metric_comparison.png"
    _save_bar_chart(
        ["Alert F1", "Break F1", "Break AUC"],
        [
            float(current["mlp"]["metrics"]["binary_alert"]["f1"]),
            float(current["mlp"]["metrics"]["binary_break"]["f1"]),
            float(current["mlp"]["metrics"]["binary_break"]["roc_auc"]),
        ],
        [
            float(reference["mlp"]["metrics"]["binary_alert"]["f1"]),
            float(reference["mlp"]["metrics"]["binary_break"]["f1"]),
            float(reference["mlp"]["metrics"]["binary_break"]["roc_auc"]),
        ],
        f"MLP 비교 ({package_dir.name} vs {reference_name})",
        p,
    )
    output.append(str(p.resolve()))

    p = visuals_dir / "hard2_axis_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, axis in zip(axes, ("x", "y", "z")):
        cur_axis = current["hard2"].get("axis_metrics", {}).get(axis, {}).get("metrics", {})
        ref_axis = reference["hard2"].get("axis_metrics", {}).get(axis, {}).get("metrics", {})
        labels = ["Best F1", "AUC", "Sep"]
        x = np.arange(3, dtype=float)
        width = 0.35
        ref_vals = [float(ref_axis.get("best_f1", 0.0)), float(ref_axis.get("auc_high_iou", 0.0)), float(ref_axis.get("separation", 0.0))]
        cur_vals = [float(cur_axis.get("best_f1", 0.0)), float(cur_axis.get("auc_high_iou", 0.0)), float(cur_axis.get("separation", 0.0))]
        ax.bar(x - width / 2, ref_vals, width, color="#9E9E9E", label="Reference")
        ax.bar(x + width / 2, cur_vals, width, color="#59A14F", label="Current")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Hard2 {axis.upper()}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    output.append(str(p.resolve()))

    p = visuals_dir / "package_metrics_overview.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(["Ref", "Current"], [float(reference["light"]["f1_score"]), float(current["light"]["f1_score"])], color=["#9E9E9E", "#4C78A8"])
    axes[0].set_title("Light F1")
    axes[1].bar(["Ref", "Current"], [float(reference["hard2"]["overall_metrics"]["best_f1"]), float(current["hard2"]["overall_metrics"]["best_f1"])], color=["#9E9E9E", "#59A14F"])
    axes[1].set_title("Hard2 Overall Best F1")
    axes[2].bar(["Ref", "Current"], [float(reference["mlp"]["metrics"]["binary_break"]["f1"]), float(current["mlp"]["metrics"]["binary_break"]["f1"])], color=["#9E9E9E", "#E15759"])
    axes[2].set_title("MLP Break F1")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
    fig.suptitle(f"{package_dir.name} vs {reference_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    output.append(str(p.resolve()))

    target_train_run = train_data_run
    if not target_train_run:
        target_train_dir = _get_latest_train_data_run()
    else:
        target_train_dir = PROJECT_ROOT / "main" / "4. make_mlp_model" / "1. mlp_train_data" / target_train_run
    if target_train_dir and target_train_dir.exists():
        x_val_path = target_train_dir / "X_val.npy"
        y_val_path = target_train_dir / "y_val.npy"
        mlp_model_path = package_dir / "mlp_model" / "model" / "mlp_pipeline.joblib"
        if x_val_path.exists() and y_val_path.exists() and mlp_model_path.exists():
            try:
                x_val = np.load(x_val_path)
                y_val = np.load(y_val_path)
                pipeline = joblib.load(mlp_model_path)
                x_val = _adapt_features_for_model(x_val, pipeline)
                val_prob = pipeline.predict_proba(x_val)[:, 1]
                rows = _threshold_sweep_rows(y_val, val_prob)
                th = np.array([r["threshold"] for r in rows], dtype=float)
                acc = np.array([r["accuracy"] for r in rows], dtype=float)
                prec = np.array([r["precision"] for r in rows], dtype=float)
                rec = np.array([r["recall"] for r in rows], dtype=float)
                f1 = np.array([r["f1"] for r in rows], dtype=float)
                suspect_th = float(current["mlp"]["thresholds"].get("suspect_threshold", 0.0))
                break_th = float(current["mlp"]["thresholds"].get("break_threshold", 1.0))

                p = visuals_dir / "mlp_threshold_sweep_detailed.png"
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(th, acc, label="Accuracy", linewidth=2, color="#1f77b4")
                ax.plot(th, prec, label="Precision (Break)", linewidth=2, color="#d62728")
                ax.plot(th, rec, label="Recall (Break)", linewidth=2, color="#2ca02c")
                ax.plot(th, f1, label="F1 (Break)", linewidth=2, color="#9467bd")
                ax.axvline(suspect_th, color="black", linestyle="--", linewidth=2, label=f"Suspect threshold={suspect_th:.4f}")
                ax.axvline(break_th, color="black", linestyle="--", linewidth=2, label=f"Break threshold={break_th:.4f}")
                ymax = max(float(acc.max()), float(prec.max()), float(rec.max()), float(f1.max()), 1.0)
                ax.text(suspect_th / 2 if suspect_th > 0 else 0.05, ymax * 0.98, "정상", ha="center", va="top", fontsize=11)
                ax.text((suspect_th + break_th) / 2, ymax * 0.98, "의심", ha="center", va="top", fontsize=11)
                ax.text((break_th + 1.0) / 2, ymax * 0.98, "파단", ha="center", va="top", fontsize=11)
                suspect_idx = int(np.argmin(np.abs(th - suspect_th)))
                break_idx = int(np.argmin(np.abs(th - break_th)))
                ax.annotate(
                    f"의심 @ {suspect_th:.4f}\nAcc {acc[suspect_idx]:.3f}\nPrec {prec[suspect_idx]:.3f}\nRec {rec[suspect_idx]:.3f}\nF1 {f1[suspect_idx]:.3f}",
                    (suspect_th, f1[suspect_idx]),
                    xytext=(10, -10),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                )
                ax.annotate(
                    f"파단 @ {break_th:.4f}\nAcc {acc[break_idx]:.3f}\nPrec {prec[break_idx]:.3f}\nRec {rec[break_idx]:.3f}\nF1 {f1[break_idx]:.3f}",
                    (break_th, f1[break_idx]),
                    xytext=(-30, -60),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                )
                ax.set_title(f"{package_dir.name} MLP Threshold Sweep")
                ax.set_xlabel("Break Threshold")
                ax.set_ylabel("Metric Value")
                ax.set_ylim(0.0, ymax * 1.02)
                ax.grid(alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                output.append(str(p.resolve()))
            except Exception as exc:
                print(f"[WARN] detailed threshold image 생성 실패: {exc}")
    return output


def _write_summary(package_dir: Path, reference_name: str, comparison: dict[str, Any]) -> None:
    lines = [
        "패키지 비교 분석",
        f"- 현재 패키지: {package_dir.name}",
        f"- 비교 기준: {reference_name}",
        "",
        "Light",
        f"- F1 변화: {comparison['light']['f1_score']['delta']:+.4f}" if comparison["light"]["f1_score"]["delta"] is not None else "- 비교 불가",
        f"- Recall 변화: {comparison['light']['recall']['delta']:+.4f}" if comparison["light"]["recall"]["delta"] is not None else "- 비교 불가",
        "",
        "Hard2",
        f"- Overall Best F1 변화: {comparison['hard2']['overall_best_f1']['delta']:+.4f}" if comparison["hard2"]["overall_best_f1"]["delta"] is not None else "- 비교 불가",
        f"- Overall AUC 변화: {comparison['hard2']['overall_auc_high_iou']['delta']:+.4f}" if comparison["hard2"]["overall_auc_high_iou"]["delta"] is not None else "- 비교 불가",
        "",
        "MLP",
        f"- Alert F1 변화: {comparison['mlp']['binary_alert_f1']['delta']:+.4f}" if comparison["mlp"]["binary_alert_f1"]["delta"] is not None else "- 비교 불가",
        f"- Break F1 변화: {comparison['mlp']['binary_break_f1']['delta']:+.4f}" if comparison["mlp"]["binary_break_f1"]["delta"] is not None else "- 비교 불가",
        f"- Break AUC 변화: {comparison['mlp']['binary_break_auc']['delta']:+.4f}" if comparison["mlp"]["binary_break_auc"]["delta"] is not None else "- 비교 불가",
    ]
    (package_dir / "패키지 비교 분석.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="기존 모델 패키지에 비교/시각화 자료를 추가합니다.")
    parser.add_argument("--package", required=True, help="대상 패키지 이름")
    parser.add_argument("--reference", required=True, help="비교 기준 패키지 이름")
    parser.add_argument("--train-data-run", default=None, help="상세 threshold 그래프 생성에 사용할 mlp train data run")
    args = parser.parse_args()

    package_dir = CURRENT_DIR / args.package
    reference_dir = CURRENT_DIR / args.reference
    if not package_dir.exists():
        raise FileNotFoundError(package_dir)
    if not reference_dir.exists():
        raise FileNotFoundError(reference_dir)

    current = _load_package_metrics(package_dir)
    reference = _load_package_metrics(reference_dir)
    comparison = _build_comparison(current, reference, args.reference)
    visuals = _generate_visuals(package_dir, current, reference, args.reference, train_data_run=args.train_data_run)
    _write_summary(package_dir, args.reference, comparison)
    (package_dir / "comparison_to_reference.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = package_dir / "package_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {"package_name": args.package}
    manifest["reference_comparison"] = {
        "reference": args.reference,
        "comparison_file": str((package_dir / "comparison_to_reference.json").resolve()),
    }
    manifest["visualization"] = {
        "generated_files": visuals,
        "overview_image": visuals[-1] if visuals else None,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 패키지 보강 완료: {package_dir}")


if __name__ == "__main__":
    main()
