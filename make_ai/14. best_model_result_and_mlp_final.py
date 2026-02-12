#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train final MLP decision model from light-best + hard2-best test outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent


def configure_tf_runtime(device: str) -> None:
    req = (device or "auto").lower()
    gpus = tf.config.list_physical_devices("GPU")

    if req == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[TF] device=cpu (GPU disabled)")
        return

    if req == "gpu":
        if not gpus:
            raise RuntimeError("device=gpu requested, but no GPU is available.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[TF] device=gpu (GPU count={len(gpus)})")
        return

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[TF] device=auto -> GPU (count={len(gpus)})")
    else:
        print("[TF] device=auto -> CPU")


def get_latest_hard_data_run(base: Path) -> Path:
    candidates: List[Path] = []
    if not base.exists():
        raise FileNotFoundError(f"Hard data base not found: {base}")
    for d in base.iterdir():
        if not d.is_dir():
            continue
        x = d / "test" / "break_imgs_test.npy"
        y = d / "test" / "break_labels_test.npy"
        if x.exists() and y.exists():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No valid hard data run found in: {base}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def flatten_binary_output(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.astype(np.float32)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0].astype(np.float32)
    return pred.reshape(pred.shape[0], -1)[:, 0].astype(np.float32)


def flatten_axis_conf(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.astype(np.float32)
    if pred.ndim == 2:
        return np.max(pred, axis=1).astype(np.float32)
    return pred.reshape(pred.shape[0], -1).max(axis=1).astype(np.float32)


def collect_test_csv_paths(run_dir: Path, n_test: int) -> List[str]:
    meta_path = run_dir / "break_imgs_metadata.json"
    if not meta_path.exists():
        return ["" for _ in range(n_test)]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        samples = meta.get("samples", [])
        test_indices = meta.get("test_indices", [])
        out: List[str] = []
        for idx in test_indices:
            if 0 <= idx < len(samples):
                out.append(str(samples[idx].get("csv_path", "")))
            else:
                out.append("")
        if len(out) == n_test:
            return out
    except Exception:
        pass
    return ["" for _ in range(n_test)]


def get_threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def find_threshold_for_recall(y_true: np.ndarray, score: np.ndarray, target_recall: float) -> Dict[str, float]:
    thresholds = np.unique(np.round(score, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    thresholds = np.unique(thresholds)
    rows = []
    for t in thresholds:
        m = get_threshold_metrics(y_true, score, float(t))
        rows.append(m)
    ok = [r for r in rows if r["recall"] >= target_recall]
    if ok:
        best = max(ok, key=lambda r: r["threshold"])
        best["reason"] = "meet_target_recall"
        return best
    best = max(rows, key=lambda r: (r["recall"], r["f1"], -abs(r["threshold"] - 0.5)))
    best["reason"] = "fallback_max_recall"
    return best


def find_threshold_for_precision(y_true: np.ndarray, score: np.ndarray, target_precision: float) -> Dict[str, float]:
    thresholds = np.unique(np.round(score, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    thresholds = np.unique(thresholds)
    rows = []
    for t in thresholds:
        m = get_threshold_metrics(y_true, score, float(t))
        rows.append(m)
    ok = [r for r in rows if r["precision"] >= target_precision]
    if ok:
        best = max(ok, key=lambda r: (r["recall"], r["precision"], r["threshold"]))
        best["reason"] = "meet_target_precision"
        return best
    best = max(rows, key=lambda r: (r["precision"], r["f1"], r["threshold"]))
    best["reason"] = "fallback_max_precision"
    return best


def parse_hidden_layers(text: str) -> Tuple[int, ...]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    out = tuple(int(v) for v in vals)
    if not out:
        return (32, 16)
    return out


def _threshold_sweep_rows(y_true: np.ndarray, score: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    for t in np.linspace(0.0, 1.0, 201):
        m = get_threshold_metrics(y_true, score, float(t))
        rows.append(m)
    return rows


def _plot_confusion_heatmap(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_eval_plots(
    out_dir: Path,
    y_true: np.ndarray,
    y_val: np.ndarray,
    all_prob: np.ndarray,
    val_prob: np.ndarray,
    pred_alert: np.ndarray,
    pred_break: np.ndarray,
    light_prob: np.ndarray,
    hard_score: np.ndarray,
    suspect_th: float,
    break_th: float,
) -> List[str]:
    image_paths: List[str] = []

    # 1) threshold sweep (validation set)
    rows = _threshold_sweep_rows(y_val, val_prob)
    th = np.array([r["threshold"] for r in rows], dtype=float)
    pr = np.array([r["precision"] for r in rows], dtype=float)
    rc = np.array([r["recall"] for r in rows], dtype=float)
    f1 = np.array([r["f1"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(th, pr, label="precision", linewidth=2)
    ax.plot(th, rc, label="recall", linewidth=2)
    ax.plot(th, f1, label="f1", linewidth=2)
    ax.axvline(suspect_th, color="orange", linestyle="--", linewidth=1.5, label=f"suspect_th={suspect_th:.3f}")
    ax.axvline(break_th, color="red", linestyle="--", linewidth=1.5, label=f"break_th={break_th:.3f}")
    ax.set_title("Validation Threshold Sweep (MLP)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "mlp_threshold_sweep_val.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    image_paths.append(str(p))

    # 2) confusion matrices
    cm_alert = confusion_matrix(y_true, pred_alert)
    cm_break = confusion_matrix(y_true, pred_break)
    p_alert = out_dir / "mlp_confusion_alert_binary.png"
    p_break = out_dir / "mlp_confusion_break_binary.png"
    _plot_confusion_heatmap(cm_alert, ["normal", "break"], "Alert Binary Confusion", p_alert)
    _plot_confusion_heatmap(cm_break, ["normal", "break"], "Break Binary Confusion", p_break)
    image_paths.extend([str(p_alert), str(p_break)])

    # 3) score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_prob[y_true == 0], bins=40, alpha=0.7, label="true normal", color="#4C78A8")
    ax.hist(all_prob[y_true == 1], bins=40, alpha=0.7, label="true break", color="#F58518")
    ax.axvline(suspect_th, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(break_th, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("MLP Break Probability Distribution")
    ax.set_xlabel("Predicted break probability")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "mlp_prob_distribution.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    image_paths.append(str(p))

    # 4) light vs hard scatter
    fig, ax = plt.subplots(figsize=(6.5, 6))
    normal_idx = y_true == 0
    break_idx = y_true == 1
    ax.scatter(light_prob[normal_idx], hard_score[normal_idx], s=14, alpha=0.45, label="true normal")
    ax.scatter(light_prob[break_idx], hard_score[break_idx], s=18, alpha=0.6, label="true break")
    ax.set_title("Light Score vs Hard Score")
    ax.set_xlabel("light_prob")
    ax.set_ylabel("hard_score")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "light_vs_hard_scatter.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    image_paths.append(str(p))

    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final MLP decision model (normal/suspect/break).")
    parser.add_argument("--hard-data-run", default=None, help="run name under 9. hard_train_data (default: latest)")
    parser.add_argument("--light-model-dir", default="light_model_best", help="light best model directory")
    parser.add_argument("--hard2-model-dir", default="hard_model_2nd_best", help="hard2 best model directory")
    parser.add_argument(
        "--output-dir",
        default="14. best_model_result_and_mlp_final",
        help="output base directory",
    )
    parser.add_argument("--run-name", default=None, help="output run name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-hidden-layers", default="32,16")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-max-iter", type=int, default=1200)
    parser.add_argument("--val-size", type=float, default=0.3)
    parser.add_argument("--weight-light", type=float, default=0.55, help="light score weight")
    parser.add_argument("--weight-hard", type=float, default=0.45, help="hard score weight")
    parser.add_argument("--weight-x", type=float, default=0.34, help="hard x axis weight")
    parser.add_argument("--weight-y", type=float, default=0.33, help="hard y axis weight")
    parser.add_argument("--weight-z", type=float, default=0.33, help="hard z axis weight")
    parser.add_argument("--target-suspect-recall", type=float, default=0.90)
    parser.add_argument("--target-break-precision", type=float, default=0.90)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="TensorFlow runtime device",
    )
    args = parser.parse_args()

    configure_tf_runtime(args.device)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    hard_data_base = CURRENT_DIR / "9. hard_train_data"
    if args.hard_data_run:
        hard_data_run = hard_data_base / args.hard_data_run
    else:
        hard_data_run = get_latest_hard_data_run(hard_data_base)
    x_test_path = hard_data_run / "test" / "break_imgs_test.npy"
    y_test_path = hard_data_run / "test" / "break_labels_test.npy"
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"Missing hard test data: {x_test_path}, {y_test_path}")

    X_test = np.load(x_test_path).astype(np.float32)
    y_test_raw = np.load(y_test_path)
    y_true = y_test_raw[:, 0].astype(np.int32) if y_test_raw.ndim == 2 else y_test_raw.astype(np.int32)
    csv_paths = collect_test_csv_paths(hard_data_run, len(X_test))
    sample_ids = [Path(p).name.replace("_OUT_processed.csv", "") if p else f"idx_{i}" for i, p in enumerate(csv_paths)]

    light_model_path = CURRENT_DIR / args.light_model_dir / "checkpoints" / "best.keras"
    hard2_dir = CURRENT_DIR / args.hard2_model_dir / "checkpoints"
    conf_x_path = hard2_dir / "conf_x.keras"
    conf_y_path = hard2_dir / "conf_y.keras"
    conf_z_path = hard2_dir / "conf_z.keras"
    for p in [light_model_path, conf_x_path, conf_y_path, conf_z_path]:
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

    print("Loading models...")
    light_model = tf.keras.models.load_model(light_model_path, compile=False)
    conf_x_model = tf.keras.models.load_model(conf_x_path, compile=False)
    conf_y_model = tf.keras.models.load_model(conf_y_path, compile=False)
    conf_z_model = tf.keras.models.load_model(conf_z_path, compile=False)

    print("Predicting on hard test set...")
    light_prob = flatten_binary_output(light_model.predict(X_test, verbose=0))
    conf_x = flatten_axis_conf(conf_x_model.predict(X_test, verbose=0))
    conf_y = flatten_axis_conf(conf_y_model.predict(X_test, verbose=0))
    conf_z = flatten_axis_conf(conf_z_model.predict(X_test, verbose=0))

    axis_weight_sum = args.weight_x + args.weight_y + args.weight_z
    if axis_weight_sum <= 0:
        raise ValueError("Axis weights sum must be > 0")
    wx = args.weight_x / axis_weight_sum
    wy = args.weight_y / axis_weight_sum
    wz = args.weight_z / axis_weight_sum

    hard_score = (wx * conf_x) + (wy * conf_y) + (wz * conf_z)
    fused_score = (args.weight_light * light_prob) + (args.weight_hard * hard_score)
    abs_gap = np.abs(light_prob - hard_score)

    features = np.column_stack(
        [
            light_prob,
            conf_x,
            conf_y,
            conf_z,
            hard_score,
            fused_score,
            abs_gap,
        ]
    ).astype(np.float32)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        features,
        y_true,
        np.arange(len(y_true)),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_true,
    )

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=parse_hidden_layers(args.mlp_hidden_layers),
                    activation="relu",
                    alpha=args.mlp_alpha,
                    max_iter=args.mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=25,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)

    val_prob = mlp.predict_proba(X_val)[:, 1]
    all_prob = mlp.predict_proba(features)[:, 1]

    suspect_rule = find_threshold_for_recall(y_val, val_prob, args.target_suspect_recall)
    break_rule = find_threshold_for_precision(y_val, val_prob, args.target_break_precision)
    suspect_th = float(suspect_rule["threshold"])
    break_th = float(break_rule["threshold"])
    if break_th < suspect_th:
        break_th = min(1.0, suspect_th + 1e-6)

    pred_break = (all_prob >= break_th).astype(np.int32)
    pred_alert = (all_prob >= suspect_th).astype(np.int32)
    pred_3cls = np.where(all_prob >= break_th, 2, np.where(all_prob >= suspect_th, 1, 0)).astype(np.int32)

    cm_binary_break = confusion_matrix(y_true, pred_break).tolist()
    cm_binary_alert = confusion_matrix(y_true, pred_alert).tolist()
    cm_3cls_vs_binary = pd.crosstab(
        pd.Series(y_true, name="true_binary"),
        pd.Series(pred_3cls, name="pred_3class"),
        dropna=False,
    )

    out_base = CURRENT_DIR / args.output_dir
    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(mlp, out_dir / "mlp_pipeline.joblib")

    pred_df = pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "sample_id": sample_ids,
            "csv_path": csv_paths,
            "y_true": y_true,
            "light_prob": light_prob,
            "hard_conf_x": conf_x,
            "hard_conf_y": conf_y,
            "hard_conf_z": conf_z,
            "hard_score": hard_score,
            "fused_score": fused_score,
            "mlp_prob_break": all_prob,
            "pred_binary_alert": pred_alert,
            "pred_binary_break": pred_break,
            "pred_3class": pred_3cls,
            "pred_3class_name": np.where(pred_3cls == 2, "break", np.where(pred_3cls == 1, "suspect", "normal")),
        }
    )
    pred_df.to_csv(out_dir / "mlp_predictions.csv", index=False, encoding="utf-8-sig")
    cm_3cls_vs_binary.to_csv(out_dir / "mlp_confusion_3class_vs_binary.csv", encoding="utf-8-sig")
    image_paths = _make_eval_plots(
        out_dir=out_dir,
        y_true=y_true,
        y_val=y_val,
        all_prob=all_prob,
        val_prob=val_prob,
        pred_alert=pred_alert,
        pred_break=pred_break,
        light_prob=light_prob,
        hard_score=hard_score,
        suspect_th=suspect_th,
        break_th=break_th,
    )

    summary = {
        "run_name": run_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "hard_data_run": hard_data_run.name,
            "light_model": str(light_model_path),
            "hard2_conf_x": str(conf_x_path),
            "hard2_conf_y": str(conf_y_path),
            "hard2_conf_z": str(conf_z_path),
        },
        "weights": {
            "light": args.weight_light,
            "hard": args.weight_hard,
            "axis_x": wx,
            "axis_y": wy,
            "axis_z": wz,
        },
        "threshold_targets": {
            "suspect_recall_target": args.target_suspect_recall,
            "break_precision_target": args.target_break_precision,
        },
        "thresholds": {
            "suspect_threshold": suspect_th,
            "break_threshold": break_th,
            "suspect_rule": suspect_rule,
            "break_rule": break_rule,
        },
        "metrics": {
            "binary_alert": {
                "precision": float(precision_score(y_true, pred_alert, zero_division=0)),
                "recall": float(recall_score(y_true, pred_alert, zero_division=0)),
                "f1": float(f1_score(y_true, pred_alert, zero_division=0)),
                "confusion_matrix": cm_binary_alert,
            },
            "binary_break": {
                "precision": float(precision_score(y_true, pred_break, zero_division=0)),
                "recall": float(recall_score(y_true, pred_break, zero_division=0)),
                "f1": float(f1_score(y_true, pred_break, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, all_prob)),
                "confusion_matrix": cm_binary_break,
            },
            "three_class_distribution": {
                "normal": int((pred_3cls == 0).sum()),
                "suspect": int((pred_3cls == 1).sum()),
                "break": int((pred_3cls == 2).sum()),
            },
        },
        "image_files": image_paths,
    }
    (out_dir / "mlp_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_lines = [
        f"run_name: {run_name}",
        f"hard_data_run: {hard_data_run.name}",
        f"suspect_threshold: {suspect_th:.6f}",
        f"break_threshold: {break_th:.6f}",
        "",
        "[binary alert (normal vs suspect+break)]",
        f"precision: {summary['metrics']['binary_alert']['precision']:.6f}",
        f"recall: {summary['metrics']['binary_alert']['recall']:.6f}",
        f"f1: {summary['metrics']['binary_alert']['f1']:.6f}",
        "",
        "[binary break (normal+suspect vs break)]",
        f"precision: {summary['metrics']['binary_break']['precision']:.6f}",
        f"recall: {summary['metrics']['binary_break']['recall']:.6f}",
        f"f1: {summary['metrics']['binary_break']['f1']:.6f}",
        f"roc_auc: {summary['metrics']['binary_break']['roc_auc']:.6f}",
    ]
    (out_dir / "mlp_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print("=" * 70)
    print(f"Saved MLP artifacts: {out_dir}")
    print(f"- {out_dir / 'mlp_pipeline.joblib'}")
    print(f"- {out_dir / 'mlp_predictions.csv'}")
    print(f"- {out_dir / 'mlp_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
