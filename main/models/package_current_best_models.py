#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
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
MAIN_BEST_DIR = PROJECT_ROOT / "main" / "best_model"
LIGHT_BEST_SELECTION = PROJECT_ROOT / "main" / "2. make_light_model" / "best_light_model" / "best_model_selection.json"
HARD2_BEST_SELECTION = PROJECT_ROOT / "main" / "3. make_hard_model" / "best_hard_model_2nd" / "overall_best" / "best_model_selection.json"
MLP_BEST_SELECTION = PROJECT_ROOT / "main" / "4. make_mlp_model" / "best_model" / "overall_best" / "best_model_selection.json"


def _to_windows_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper() + ":"
        rest = text[6:].replace("/", "\\")
        return Path(drive + rest)
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
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


def collect_current_sources() -> dict[str, Any]:
    light_selection = _load_json(LIGHT_BEST_SELECTION)
    hard2_selection = _load_json(HARD2_BEST_SELECTION)
    mlp_selection = _load_json(MLP_BEST_SELECTION)
    dependency_info = _load_json(MAIN_BEST_DIR / "model_dependency_info.json")

    light_report_path = _to_windows_path(light_selection["selected"]["report_path"])
    light_eval = _load_json(light_report_path)
    light_run = str(light_selection["selected"]["model_run"])
    light_run_dir = PROJECT_ROOT / "main" / "2. make_light_model" / "2. light_models" / light_run

    hard2_run = str(hard2_selection["selected"]["model_run"])
    hard2_run_dir = PROJECT_ROOT / "main" / "3. make_hard_model" / "3. hard_models_2nd" / hard2_run
    hard2_eval_path = hard2_run_dir / "evaluate" / "evaluation_metrics.json"
    hard2_eval = _load_json(hard2_eval_path)

    mlp_run = str(mlp_selection["selected"]["model_run"])
    mlp_run_dir = PROJECT_ROOT / "main" / "4. make_mlp_model" / "2. mlp_models" / mlp_run
    mlp_eval_path = mlp_run_dir / "training_results.json"
    mlp_eval = _load_json(mlp_eval_path)

    return {
        "dependency_info": dependency_info,
        "light": {
            "selection": light_selection,
            "report": light_eval,
            "run_dir": light_run_dir,
            "run_name": light_run,
            "evaluation_dir": light_run_dir / "evaluation",
        },
        "hard2": {
            "selection": hard2_selection,
            "report": hard2_eval,
            "run_dir": hard2_run_dir,
            "run_name": hard2_run,
            "evaluation_dir": hard2_run_dir / "evaluate",
        },
        "mlp": {
            "selection": mlp_selection,
            "report": mlp_eval,
            "run_dir": mlp_run_dir,
            "run_name": mlp_run,
        },
    }


def build_light_metrics(light_report: dict[str, Any]) -> dict[str, Any]:
    classification = light_report["classification"]
    return {
        "accuracy": classification["accuracy"],
        "precision": classification["precision"],
        "recall": classification["recall"],
        "f1_score": classification["f1_score"],
        "roc_auc": classification["roc_auc"],
        "pr_auc": classification["pr_auc"],
        "confusion_matrix": classification["confusion_matrix"],
        "confusion_matrix_normalized": classification["confusion_matrix_normalized"],
        "all_data_metrics": classification.get("all_data_metrics", {}),
    }


def build_hard2_metrics(selection: dict[str, Any], hard2_report: dict[str, Any]) -> dict[str, Any]:
    axis_metrics: dict[str, Any] = {}
    for axis, axis_info in hard2_report.get("by_axis", {}).items():
        axis_metrics[axis] = {
            "run": selection["selected"]["model_run"],
            "metrics": axis_info,
        }
    return {
        "selection": {
            "model_run": selection["selected"]["model_run"],
            "first_stage_run": selection["selected"].get("first_stage_run"),
            "selected_at": selection.get("updated_at"),
            "strategy": "overall_best_single_run",
        },
        "axis_metrics": axis_metrics,
        "overall_metrics": hard2_report.get("overall", {}),
    }


def build_mlp_summary(mlp_report: dict[str, Any]) -> dict[str, Any]:
    metadata = mlp_report.get("train_data_metadata", {})
    return {
        "run_name": mlp_report["run_name"],
        "created_at": mlp_report["created_at"],
        "train_data_run": mlp_report.get("train_data_run", ""),
        "sources": {
            "hard_data_run": Path(str(metadata.get("hard_data_run", ""))).name if metadata.get("hard_data_run") else "",
            "light_model": metadata.get("light_model_path", ""),
            "hard1_model": metadata.get("hard1_model_dir", ""),
            "hard2_model": metadata.get("hard2_model_dir", ""),
        },
        "weights": metadata.get("weights", {}),
        "threshold_targets": {
            "suspect_recall_target": metadata.get("target_suspect_recall"),
            "break_precision_target": metadata.get("target_break_precision"),
        },
        "thresholds": mlp_report.get("metrics", {}).get("thresholds", {}),
        "metrics": {
            "binary_alert": mlp_report.get("metrics", {}).get("binary_alert", {}),
            "binary_break": mlp_report.get("metrics", {}).get("binary_break", {}),
            "three_class_distribution": {
                "normal": metadata.get("class_distribution", {}).get("normal"),
                "break": metadata.get("class_distribution", {}).get("break"),
            },
        },
        "passed": mlp_report.get("passed"),
        "feedback": mlp_report.get("feedback"),
        "model_config": mlp_report.get("model_config", {}),
    }


def load_previous_metrics(compare_dir: Path) -> dict[str, Any]:
    hard2_prev = _load_json(compare_dir / "hard_model_2nd" / "test_results" / "hard2_axis_test_metrics.json")
    if "overall_metrics" not in hard2_prev:
        axis_metrics = hard2_prev.get("axis_metrics", {})
        overall_best_f1 = np.mean([float(axis_metrics[a]["metrics"].get("best_f1", 0.0)) for a in axis_metrics]) if axis_metrics else None
        overall_auc = np.mean([float(axis_metrics[a]["metrics"].get("auc_high_iou", 0.0)) for a in axis_metrics]) if axis_metrics else None
        overall_sep = np.mean([float(axis_metrics[a]["metrics"].get("separation", 0.0)) for a in axis_metrics]) if axis_metrics else None
        hard2_prev["overall_metrics"] = {
            "best_f1": overall_best_f1,
            "auc_high_iou": overall_auc,
            "separation": overall_sep,
        }
    return {
        "light": _load_json(compare_dir / "light_model" / "test_results" / "light_test_metrics.json"),
        "hard2": hard2_prev,
        "mlp": _load_json(compare_dir / "mlp_model" / "test_results" / "mlp_summary.json"),
    }


def build_comparison(current: dict[str, Any], previous: dict[str, Any], previous_name: str) -> dict[str, Any]:
    current_light = current["light"]
    prev_light = previous["light"]

    current_hard2 = current["hard2"]
    prev_hard2 = previous["hard2"]

    current_mlp = current["mlp"]
    prev_mlp = previous["mlp"]

    axis_comparison = {}
    for axis in ("x", "y", "z"):
        cur_axis = current_hard2["axis_metrics"].get(axis, {}).get("metrics", {})
        prev_axis = prev_hard2["axis_metrics"].get(axis, {}).get("metrics", {})
        axis_comparison[axis] = {
            "best_f1": _metric_delta(_safe_float(cur_axis.get("best_f1")), _safe_float(prev_axis.get("best_f1"))),
            "auc_high_iou": _metric_delta(_safe_float(cur_axis.get("auc_high_iou")), _safe_float(prev_axis.get("auc_high_iou"))),
            "separation": _metric_delta(_safe_float(cur_axis.get("separation")), _safe_float(prev_axis.get("separation"))),
        }

    return {
        "compare_to": previous_name,
        "light": {
            "f1_score": _metric_delta(_safe_float(current_light.get("f1_score")), _safe_float(prev_light.get("f1_score"))),
            "recall": _metric_delta(_safe_float(current_light.get("recall")), _safe_float(prev_light.get("recall"))),
            "precision": _metric_delta(_safe_float(current_light.get("precision")), _safe_float(prev_light.get("precision"))),
            "roc_auc": _metric_delta(_safe_float(current_light.get("roc_auc")), _safe_float(prev_light.get("roc_auc"))),
        },
        "hard2": {
            "overall_best_f1": _metric_delta(
                _safe_float(current_hard2.get("overall_metrics", {}).get("best_f1")),
                _safe_float(prev_hard2.get("overall_metrics", {}).get("best_f1")),
            ),
            "overall_auc_high_iou": _metric_delta(
                _safe_float(current_hard2.get("overall_metrics", {}).get("auc_high_iou")),
                _safe_float(prev_hard2.get("overall_metrics", {}).get("auc_high_iou")),
            ),
            "overall_separation": _metric_delta(
                _safe_float(current_hard2.get("overall_metrics", {}).get("separation")),
                _safe_float(prev_hard2.get("overall_metrics", {}).get("separation")),
            ),
            "by_axis": axis_comparison,
        },
        "mlp": {
            "binary_alert_f1": _metric_delta(
                _safe_float(current_mlp.get("metrics", {}).get("binary_alert", {}).get("f1")),
                _safe_float(prev_mlp.get("metrics", {}).get("binary_alert", {}).get("f1")),
            ),
            "binary_break_f1": _metric_delta(
                _safe_float(current_mlp.get("metrics", {}).get("binary_break", {}).get("f1")),
                _safe_float(prev_mlp.get("metrics", {}).get("binary_break", {}).get("f1")),
            ),
            "binary_break_auc": _metric_delta(
                _safe_float(current_mlp.get("metrics", {}).get("binary_break", {}).get("roc_auc")),
                _safe_float(prev_mlp.get("metrics", {}).get("binary_break", {}).get("roc_auc")),
            ),
        },
    }


def write_summary_text(package_dir: Path, package_name: str, manifest: dict[str, Any], current: dict[str, Any], comparison: dict[str, Any]) -> None:
    light = current["light"]
    hard2 = current["hard2"]
    mlp = current["mlp"]
    compare_to = comparison["compare_to"]

    lines = [
        "모델 성능 평가",
        "(자동 생성 요약)",
        "",
        "패키지 정보",
        f"- 패키지 이름: {package_name}",
        f"- 생성 시각: {manifest['created_at']}",
        f"- 비교 기준: {compare_to}",
        "",
        "모델 산출물 위치(패키지 기준)",
        "- Light 모델: light_model/saved_model",
        "- Hard 2차 모델: hard_model_2nd/saved_model",
        "- MLP 모델: mlp_model/model/mlp_pipeline.joblib",
        "",
        "1. Light 모델 성능",
        f"- Run: {light['selection']['selected']['model_run']}",
        f"- F1: {light['metrics']['f1_score']:.4f}",
        f"- Recall: {light['metrics']['recall']:.4f}",
        f"- Precision: {light['metrics']['precision']:.4f}",
        f"- ROC AUC: {light['metrics']['roc_auc']:.4f}",
        f"- {compare_to} 대비 F1 변화: {comparison['light']['f1_score']['delta']:+.4f}" if comparison['light']['f1_score']['delta'] is not None else "- 비교 불가",
        "",
        "2. Hard 2차 모델 성능",
        f"- Run: {hard2['selection']['selection']['model_run']}",
        f"- Overall Best F1: {hard2['metrics']['overall_metrics'].get('best_f1', 0):.4f}",
        f"- Overall AUC(high IoU): {hard2['metrics']['overall_metrics'].get('auc_high_iou', 0):.4f}",
        f"- Overall Separation: {hard2['metrics']['overall_metrics'].get('separation', 0):.4f}",
        f"- {compare_to} 대비 Overall Best F1 변화: {comparison['hard2']['overall_best_f1']['delta']:+.4f}" if comparison['hard2']['overall_best_f1']['delta'] is not None else "- 비교 불가",
        "",
        "3. MLP 융합 모델 성능",
        f"- Run: {mlp['summary']['run_name']}",
        f"- Binary Alert F1: {mlp['summary']['metrics']['binary_alert'].get('f1', 0):.4f}",
        f"- Binary Break F1: {mlp['summary']['metrics']['binary_break'].get('f1', 0):.4f}",
        f"- Binary Break AUC: {mlp['summary']['metrics']['binary_break'].get('roc_auc', 0):.4f}",
        f"- {compare_to} 대비 Binary Break F1 변화: {comparison['mlp']['binary_break_f1']['delta']:+.4f}" if comparison['mlp']['binary_break_f1']['delta'] is not None else "- 비교 불가",
        "",
        "세부 비교 파일",
        "- comparison_to_previous.json",
        "- light_model/test_results/light_test_metrics.json",
        "- hard_model_2nd/test_results/hard2_axis_test_metrics.json",
        "- mlp_model/test_results/mlp_summary.json",
        "- visuals/mlp_threshold_sweep_detailed.png",
        "- visuals/package_metrics_overview.png",
        "- visuals/light_metric_comparison.png",
        "- visuals/hard2_axis_comparison.png",
        "- visuals/mlp_metric_comparison.png",
    ]
    (package_dir / "모델 성능 평가.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_bar_comparison(labels: list[str], current_vals: list[float], previous_vals: list[float], title: str, out_path: Path) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, previous_vals, width, label="Previous", color="#9E9E9E")
    ax.bar(x + width / 2, current_vals, width, label="Current", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, max(max(current_vals), max(previous_vals), 1.0) * 1.15)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_visualizations(package_dir: Path, current: dict[str, Any], previous: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    visuals_dir = package_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []

    light_curr = current["light"]
    light_prev = previous["light"]
    light_chart = visuals_dir / "light_metric_comparison.png"
    _save_bar_comparison(
        labels=["F1", "Recall", "Precision", "ROC AUC"],
        current_vals=[
            float(light_curr["f1_score"]),
            float(light_curr["recall"]),
            float(light_curr["precision"]),
            float(light_curr["roc_auc"]),
        ],
        previous_vals=[
            float(light_prev["f1_score"]),
            float(light_prev["recall"]),
            float(light_prev["precision"]),
            float(light_prev["roc_auc"]),
        ],
        title="Light Model Metrics Comparison",
        out_path=light_chart,
    )
    output_paths.append(str(light_chart.resolve()))

    hard2_curr = current["hard2"]["overall_metrics"]
    hard2_prev = previous["hard2"].get("overall_metrics", {})
    hard2_chart = visuals_dir / "hard2_axis_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, axis in zip(axes, ("x", "y", "z")):
        cur_axis = current["hard2"]["axis_metrics"].get(axis, {}).get("metrics", {})
        prev_axis = previous["hard2"]["axis_metrics"].get(axis, {}).get("metrics", {})
        labels = ["Best F1", "AUC", "Sep"]
        prev_vals = [
            float(prev_axis.get("best_f1", 0.0)),
            float(prev_axis.get("auc_high_iou", 0.0)),
            float(prev_axis.get("separation", 0.0)),
        ]
        cur_vals = [
            float(cur_axis.get("best_f1", 0.0)),
            float(cur_axis.get("auc_high_iou", 0.0)),
            float(cur_axis.get("separation", 0.0)),
        ]
        x = np.arange(len(labels), dtype=float)
        width = 0.35
        ax.bar(x - width / 2, prev_vals, width, label="Previous", color="#9E9E9E")
        ax.bar(x + width / 2, cur_vals, width, label="Current", color="#59A14F")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Hard2 Axis {axis.upper()}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(hard2_chart, dpi=160)
    plt.close(fig)
    output_paths.append(str(hard2_chart.resolve()))

    mlp_curr = current["mlp"]
    mlp_prev = previous["mlp"]
    mlp_chart = visuals_dir / "mlp_metric_comparison.png"
    _save_bar_comparison(
        labels=["Alert F1", "Break F1", "Break AUC"],
        current_vals=[
            float(mlp_curr["metrics"]["binary_alert"]["f1"]),
            float(mlp_curr["metrics"]["binary_break"]["f1"]),
            float(mlp_curr["metrics"]["binary_break"]["roc_auc"]),
        ],
        previous_vals=[
            float(mlp_prev["metrics"]["binary_alert"]["f1"]),
            float(mlp_prev["metrics"]["binary_break"]["f1"]),
            float(mlp_prev["metrics"]["binary_break"]["roc_auc"]),
        ],
        title="MLP Metrics Comparison",
        out_path=mlp_chart,
    )
    output_paths.append(str(mlp_chart.resolve()))

    overview = visuals_dir / "package_metrics_overview.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(["Prev", "Current"], [float(light_prev["f1_score"]), float(light_curr["f1_score"])], color=["#9E9E9E", "#4C78A8"])
    axes[0].set_title("Light F1")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        ["Prev", "Current"],
        [float(previous["hard2"]["axis_metrics"]["z"]["metrics"]["best_f1"]), float(current["hard2"]["axis_metrics"]["z"]["metrics"]["best_f1"])],
        color=["#9E9E9E", "#59A14F"],
    )
    axes[1].set_title("Hard2 Z-axis Best F1")
    axes[1].set_ylim(0, max(1.0, float(previous["hard2"]["axis_metrics"]["z"]["metrics"]["best_f1"]), float(current["hard2"]["axis_metrics"]["z"]["metrics"]["best_f1"])) * 1.15)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(
        ["Prev", "Current"],
        [float(mlp_prev["metrics"]["binary_break"]["f1"]), float(mlp_curr["metrics"]["binary_break"]["f1"])],
        color=["#9E9E9E", "#E15759"],
    )
    axes[2].set_title("MLP Break F1")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Model Package Comparison: {comparison['compare_to']} vs Current", fontsize=13)
    fig.tight_layout()
    fig.savefig(overview, dpi=160)
    plt.close(fig)
    output_paths.append(str(overview.resolve()))

    mlp_train_run = str(current["mlp"].get("train_data_run", ""))
    if mlp_train_run:
        train_data_dir = PROJECT_ROOT / "main" / "4. make_mlp_model" / "1. mlp_train_data" / mlp_train_run
        x_val_path = train_data_dir / "X_val.npy"
        y_val_path = train_data_dir / "y_val.npy"
        mlp_model_path = package_dir / "mlp_model" / "model" / "mlp_pipeline.joblib"
        if x_val_path.exists() and y_val_path.exists() and mlp_model_path.exists():
            try:
                x_val = np.load(x_val_path)
                y_val = np.load(y_val_path)
                pipeline = joblib.load(mlp_model_path)
                val_prob = pipeline.predict_proba(x_val)[:, 1]
                rows = _threshold_sweep_rows(y_val, val_prob)
                th = np.array([r["threshold"] for r in rows], dtype=float)
                acc = np.array([r["accuracy"] for r in rows], dtype=float)
                prec = np.array([r["precision"] for r in rows], dtype=float)
                rec = np.array([r["recall"] for r in rows], dtype=float)
                f1 = np.array([r["f1"] for r in rows], dtype=float)
                suspect_th = float(current["mlp"]["thresholds"].get("suspect_threshold", 0.0))
                break_th = float(current["mlp"]["thresholds"].get("break_threshold", 1.0))

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
                ax.scatter([suspect_th], [f1[suspect_idx]], color="#9467bd", zorder=5)
                ax.scatter([break_th], [f1[break_idx]], color="#9467bd", zorder=5)
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
                detailed_path = visuals_dir / "mlp_threshold_sweep_detailed.png"
                fig.savefig(detailed_path, dpi=160)
                plt.close(fig)
                output_paths.append(str(detailed_path.resolve()))
            except Exception as exc:
                print(f"[WARN] detailed MLP threshold image 생성 실패: {exc}")

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="현재 베스트 모델을 main/models/<name> 패키지로 내보냅니다.")
    parser.add_argument("--name", required=True, help="패키지 폴더 이름")
    parser.add_argument("--compare-to", default="Genesis", help="비교 기준 패키지 이름")
    parser.add_argument("--force", action="store_true", help="대상 폴더가 있으면 덮어쓰기")
    args = parser.parse_args()

    package_dir = CURRENT_DIR / args.name
    compare_dir = CURRENT_DIR / args.compare_to

    if package_dir.exists():
        if not args.force:
            raise FileExistsError(f"이미 존재하는 패키지 폴더입니다: {package_dir}")
        shutil.rmtree(package_dir)

    if not compare_dir.exists():
        raise FileNotFoundError(f"비교 기준 패키지를 찾을 수 없습니다: {compare_dir}")

    current_sources = collect_current_sources()
    previous_metrics = load_previous_metrics(compare_dir)

    package_dir.mkdir(parents=True, exist_ok=True)

    light_saved_dir = package_dir / "light_model" / "saved_model"
    hard2_saved_dir = package_dir / "hard_model_2nd" / "saved_model"
    mlp_model_dir = package_dir / "mlp_model" / "model"
    light_test_dir = package_dir / "light_model" / "test_results"
    hard2_test_dir = package_dir / "hard_model_2nd" / "test_results"
    mlp_test_dir = package_dir / "mlp_model" / "test_results"

    _copy_path(MAIN_BEST_DIR / "light_model", light_saved_dir)
    _copy_path(MAIN_BEST_DIR / "hard2_model", hard2_saved_dir)
    _copy_path(MAIN_BEST_DIR / "mlp_model", mlp_model_dir)
    _copy_path(current_sources["light"]["evaluation_dir"], light_test_dir)
    light_metrics = build_light_metrics(current_sources["light"]["report"])
    (light_test_dir / "light_test_metrics.json").write_text(json.dumps(light_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    hard2_test_dir.mkdir(parents=True, exist_ok=True)
    _copy_path(current_sources["hard2"]["evaluation_dir"] / "evaluation_metrics.json", hard2_test_dir / "evaluation_metrics.json")
    _copy_path(current_sources["hard2"]["evaluation_dir"] / "training_feedback.json", hard2_test_dir / "training_feedback.json")
    hard2_metrics = build_hard2_metrics(current_sources["hard2"]["selection"], current_sources["hard2"]["report"])
    (hard2_test_dir / "hard2_axis_test_metrics.json").write_text(json.dumps(hard2_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    mlp_test_dir.mkdir(parents=True, exist_ok=True)
    _copy_path(current_sources["mlp"]["run_dir"] / "training_results.json", mlp_test_dir / "training_results.json")
    mlp_summary = build_mlp_summary(current_sources["mlp"]["report"])
    (mlp_test_dir / "mlp_summary.json").write_text(json.dumps(mlp_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    light_source_info = {
        "light_model_path": str((MAIN_BEST_DIR / "light_model" / "best.keras").resolve()),
        "selection_meta": current_sources["light"]["selection"],
    }
    hard2_source_info = {
        "conf_x": str((MAIN_BEST_DIR / "hard2_model" / "conf_x.keras").resolve()),
        "conf_y": str((MAIN_BEST_DIR / "hard2_model" / "conf_y.keras").resolve()),
        "conf_z": str((MAIN_BEST_DIR / "hard2_model" / "conf_z.keras").resolve()),
        "selection": current_sources["hard2"]["selection"]["selected"],
    }
    mlp_source_info = {
        "mlp_run_dir": str(current_sources["mlp"]["run_dir"].resolve()),
        "selection_meta": current_sources["mlp"]["selection"],
    }

    (package_dir / "light_model" / "source_info.json").write_text(json.dumps(light_source_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (package_dir / "hard_model_2nd" / "source_info.json").write_text(json.dumps(hard2_source_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (package_dir / "mlp_model" / "source_info.json").write_text(json.dumps(mlp_source_info, ensure_ascii=False, indent=2), encoding="utf-8")

    current_metrics_for_compare = {
        "light": light_metrics,
        "hard2": hard2_metrics,
        "mlp": mlp_summary,
    }
    comparison = build_comparison(current_metrics_for_compare, previous_metrics, args.compare_to)
    (package_dir / "comparison_to_previous.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    created_at = dt.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "package_name": args.name,
        "created_at": created_at,
        "compare_to": args.compare_to,
        "sources": {
            "light_run": current_sources["light"]["run_name"],
            "hard2_run": current_sources["hard2"]["run_name"],
            "mlp_run": current_sources["mlp"]["run_name"],
            "dependency_info": current_sources["dependency_info"],
        },
        "exports": {
            "light_model": {
                "saved_model_dir": str(light_saved_dir.resolve()),
                "test_metrics_file": str((light_test_dir / "light_test_metrics.json").resolve()),
            },
            "hard_model_2nd": {
                "saved_model_dir": str(hard2_saved_dir.resolve()),
                "test_metrics_file": str((hard2_test_dir / "hard2_axis_test_metrics.json").resolve()),
            },
            "mlp_model": {
                "model_file": str((mlp_model_dir / "mlp_pipeline.joblib").resolve()),
                "test_metrics_file": str((mlp_test_dir / "mlp_summary.json").resolve()),
            },
        },
        "comparison_file": str((package_dir / "comparison_to_previous.json").resolve()),
    }
    current_metrics_for_compare = {
        "light": light_metrics,
        "hard2": hard2_metrics,
        "mlp": mlp_summary,
    }
    visualization_files = generate_visualizations(package_dir, current_metrics_for_compare, previous_metrics, comparison)
    manifest["visualization"] = {
        "generated_files": visualization_files,
        "overview_image": visualization_files[-1] if visualization_files else None,
    }
    (package_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_text(
        package_dir=package_dir,
        package_name=args.name,
        manifest=manifest,
        current={
            "light": {"selection": current_sources["light"]["selection"], "metrics": light_metrics},
            "hard2": {"selection": hard2_metrics, "metrics": hard2_metrics},
            "mlp": {"summary": mlp_summary},
        },
        comparison=comparison,
    )

    print(f"[DONE] 모델 패키지 생성 완료: {package_dir}")


if __name__ == "__main__":
    main()
