#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export light/hard2/mlp best models and attach test-result artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def export_keras_saved_model(src_keras_path: Path, dst_saved_model_dir: Path) -> None:
    ensure_dir(dst_saved_model_dir.parent)
    if dst_saved_model_dir.exists():
        shutil.rmtree(dst_saved_model_dir)
    model = tf.keras.models.load_model(src_keras_path, compile=False)
    # Keras 3 prefers model.export() for SavedModel export.
    if hasattr(model, "export"):
        model.export(str(dst_saved_model_dir))
        return
    tf.saved_model.save(model, str(dst_saved_model_dir))


def find_latest_mlp_run(base_dir: Path) -> Path:
    candidates = []
    if not base_dir.exists():
        raise FileNotFoundError(f"MLP base directory not found: {base_dir}")
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "mlp_pipeline.joblib").exists() and (d / "mlp_summary.json").exists():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No MLP run with required artifacts found in: {base_dir}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def make_package_overview_plot(
    package_dir: Path,
    light_metrics: Dict,
    hard2_axis_metrics: Dict,
    mlp_summary: Dict,
) -> Optional[Path]:
    try:
        light_vals = [
            _to_float(light_metrics.get("accuracy")),
            _to_float(light_metrics.get("precision")),
            _to_float(light_metrics.get("recall")),
            _to_float(light_metrics.get("f1_score") if "f1_score" in light_metrics else light_metrics.get("f1")),
        ]

        h_x = _to_float((hard2_axis_metrics.get("x", {}).get("metrics", {}) or {}).get("best_f1"))
        h_y = _to_float((hard2_axis_metrics.get("y", {}).get("metrics", {}) or {}).get("best_f1"))
        h_z = _to_float((hard2_axis_metrics.get("z", {}).get("metrics", {}) or {}).get("best_f1"))
        hard_vals = [h_x, h_y, h_z]

        mlp_bin_alert = ((mlp_summary.get("metrics") or {}).get("binary_alert") or {})
        mlp_bin_break = ((mlp_summary.get("metrics") or {}).get("binary_break") or {})
        mlp_vals = [
            _to_float(mlp_bin_alert.get("recall")),
            _to_float(mlp_bin_break.get("precision")),
            _to_float(mlp_bin_break.get("f1")),
            _to_float(mlp_bin_break.get("roc_auc")),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        axes[0].bar(["acc", "prec", "rec", "f1"], light_vals, color=["#4C78A8"] * 4)
        axes[0].set_ylim(0, 1.0)
        axes[0].set_title("Light Test Metrics")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(["x", "y", "z"], hard_vals, color=["#59A14F", "#F28E2B", "#E15759"])
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("Hard2 Axis best_f1")
        axes[1].grid(axis="y", alpha=0.25)

        axes[2].bar(["alert_rec", "break_prec", "break_f1", "auc"], mlp_vals, color=["#B07AA1"] * 4)
        axes[2].set_ylim(0, 1.0)
        axes[2].set_title("MLP Metrics")
        axes[2].grid(axis="y", alpha=0.25)

        fig.suptitle("Saved Package Metrics Overview", fontsize=12)
        fig.tight_layout()
        out_path = package_dir / "package_metrics_overview.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3 best models to deployable package.")
    parser.add_argument("--light-dir", default="light_model_best", help="light best directory")
    parser.add_argument("--hard2-dir", default="hard_model_2nd_best", help="hard2 best directory")
    parser.add_argument(
        "--mlp-run",
        default=None,
        help="MLP run name under 14. best_model_result_and_mlp_final (default: latest)",
    )
    parser.add_argument(
        "--mlp-base-dir",
        default="14. best_model_result_and_mlp_final",
        help="MLP artifact base dir",
    )
    parser.add_argument("--output-dir", default="15. make_save_model", help="export package base dir")
    parser.add_argument("--package-name", default=None, help="package folder name")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="TensorFlow runtime device",
    )
    args = parser.parse_args()
    configure_tf_runtime(args.device)

    light_dir = CURRENT_DIR / args.light_dir
    hard2_dir = CURRENT_DIR / args.hard2_dir
    mlp_base = CURRENT_DIR / args.mlp_base_dir
    out_base = CURRENT_DIR / args.output_dir

    light_best = light_dir / "checkpoints" / "best.keras"
    conf_x = hard2_dir / "checkpoints" / "conf_x.keras"
    conf_y = hard2_dir / "checkpoints" / "conf_y.keras"
    conf_z = hard2_dir / "checkpoints" / "conf_z.keras"

    for p in [light_best, conf_x, conf_y, conf_z]:
        if not p.exists():
            raise FileNotFoundError(f"Required model file not found: {p}")

    if args.mlp_run:
        mlp_run_dir = mlp_base / args.mlp_run
    else:
        mlp_run_dir = find_latest_mlp_run(mlp_base)
    mlp_model_file = mlp_run_dir / "mlp_pipeline.joblib"
    mlp_summary_file = mlp_run_dir / "mlp_summary.json"
    if not mlp_model_file.exists() or not mlp_summary_file.exists():
        raise FileNotFoundError(f"MLP artifacts missing in: {mlp_run_dir}")

    package_name = args.package_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = out_base / package_name
    ensure_dir(package_dir)

    manifest: Dict = {
        "package_name": package_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "light_dir": str(light_dir),
            "hard2_dir": str(hard2_dir),
            "mlp_run_dir": str(mlp_run_dir),
        },
        "exports": {},
    }

    # ------------------------------------------------------------------ #
    # 1) Light model export + light test result attachment
    # ------------------------------------------------------------------ #
    light_out = package_dir / "light_model"
    light_saved = light_out / "saved_model"
    ensure_dir(light_out)
    export_keras_saved_model(light_best, light_saved)

    light_meta = load_json(light_dir / "best_model_selection.json") or {}
    selected = (light_meta or {}).get("selected", {})
    light_report_src = Path(selected.get("report_path", "")) if selected.get("report_path") else None
    light_metrics = {}
    light_visuals = []
    if light_report_src and light_report_src.exists():
        light_report_dst = light_out / "test_results" / "evaluation_report.json"
        copy_if_exists(light_report_src, light_report_dst)
        maybe_txt = light_report_src.with_name("evaluation_summary.txt")
        copy_if_exists(maybe_txt, light_out / "test_results" / "evaluation_summary.txt")
        for png in sorted(light_report_src.parent.glob("*.png")):
            dst_png = light_out / "test_results" / png.name
            if copy_if_exists(png, dst_png):
                light_visuals.append(str(dst_png))
        report_data = load_json(light_report_src)
        if report_data:
            light_metrics = report_data.get("classification", {})

    save_json(light_out / "test_results" / "light_test_metrics.json", light_metrics)
    save_json(light_out / "source_info.json", {"light_model_path": str(light_best), "selection_meta": light_meta})
    manifest["exports"]["light_model"] = {
        "saved_model_dir": str(light_saved),
        "test_metrics_file": str(light_out / "test_results" / "light_test_metrics.json"),
        "visual_files": light_visuals,
    }

    # ------------------------------------------------------------------ #
    # 2) Hard2 model export + hard2 test result attachment
    # ------------------------------------------------------------------ #
    hard2_out = package_dir / "hard_model_2nd"
    hard2_saved = hard2_out / "saved_model"
    ensure_dir(hard2_out)

    export_keras_saved_model(conf_x, hard2_saved / "conf_x")
    export_keras_saved_model(conf_y, hard2_saved / "conf_y")
    export_keras_saved_model(conf_z, hard2_saved / "conf_z")

    hard2_select = load_json(hard2_dir / "selected_from_runs.json") or {}
    eval_base = CURRENT_DIR / "13. evaluate_hard_model_2nd"
    ensure_dir(hard2_out / "test_results")
    axis_runs = {
        "x": hard2_select.get("x_run"),
        "y": hard2_select.get("y_run"),
        "z": hard2_select.get("z_run"),
    }
    hard2_axis_metrics: Dict = {}
    hard2_visuals = []
    for axis, run_name in axis_runs.items():
        if not run_name:
            continue
        summary_src = eval_base / str(run_name) / "test" / "evaluation_summary.json"
        stat_src = eval_base / str(run_name) / "test" / "confidence_statistics.json"
        axis_dir = hard2_out / "test_results" / axis
        ensure_dir(axis_dir)
        copied_summary = copy_if_exists(summary_src, axis_dir / "evaluation_summary.json")
        copy_if_exists(stat_src, axis_dir / "confidence_statistics.json")
        if copied_summary:
            d = load_json(summary_src) or {}
            by_axis = ((d.get("metrics") or {}).get("by_axis") or {}).get(axis, {})
            hard2_axis_metrics[axis] = {"run": run_name, "metrics": by_axis}
            src_run_dir = eval_base / str(run_name)
            for p in sorted((src_run_dir / "test").glob(f"*_{axis}.png")):
                dst_p = axis_dir / p.name
                if copy_if_exists(p, dst_p):
                    hard2_visuals.append(str(dst_p))
        else:
            hard2_axis_metrics[axis] = {"run": run_name, "metrics": {}}

    save_json(
        hard2_out / "test_results" / "hard2_axis_test_metrics.json",
        {
            "selection": hard2_select,
            "axis_metrics": hard2_axis_metrics,
        },
    )
    save_json(
        hard2_out / "source_info.json",
        {
            "conf_x": str(conf_x),
            "conf_y": str(conf_y),
            "conf_z": str(conf_z),
            "selection": hard2_select,
        },
    )
    manifest["exports"]["hard_model_2nd"] = {
        "saved_model_dir": str(hard2_saved),
        "test_metrics_file": str(hard2_out / "test_results" / "hard2_axis_test_metrics.json"),
        "visual_files": hard2_visuals,
    }

    # ------------------------------------------------------------------ #
    # 3) MLP artifact bundle + MLP test result attachment
    # ------------------------------------------------------------------ #
    mlp_out = package_dir / "mlp_model"
    ensure_dir(mlp_out / "model")
    ensure_dir(mlp_out / "test_results")

    # Verify model can be loaded before package finalization.
    _ = joblib.load(mlp_model_file)
    copy_if_exists(mlp_model_file, mlp_out / "model" / "mlp_pipeline.joblib")
    copy_if_exists(mlp_summary_file, mlp_out / "test_results" / "mlp_summary.json")
    copy_if_exists(mlp_run_dir / "mlp_predictions.csv", mlp_out / "test_results" / "mlp_predictions.csv")
    copy_if_exists(
        mlp_run_dir / "mlp_confusion_3class_vs_binary.csv",
        mlp_out / "test_results" / "mlp_confusion_3class_vs_binary.csv",
    )
    mlp_visuals = []
    for png in sorted(mlp_run_dir.glob("*.png")):
        dst_png = mlp_out / "test_results" / png.name
        if copy_if_exists(png, dst_png):
            mlp_visuals.append(str(dst_png))
    save_json(mlp_out / "source_info.json", {"mlp_run_dir": str(mlp_run_dir)})
    manifest["exports"]["mlp_model"] = {
        "model_file": str(mlp_out / "model" / "mlp_pipeline.joblib"),
        "test_metrics_file": str(mlp_out / "test_results" / "mlp_summary.json"),
        "visual_files": mlp_visuals,
    }

    mlp_summary = load_json(mlp_summary_file) or {}
    overview_img = make_package_overview_plot(
        package_dir=package_dir,
        light_metrics=light_metrics,
        hard2_axis_metrics=hard2_axis_metrics,
        mlp_summary=mlp_summary,
    )
    if overview_img is not None:
        manifest["visualization"] = {"overview_image": str(overview_img)}

    save_json(package_dir / "package_manifest.json", manifest)

    print("=" * 70)
    print(f"Saved package: {package_dir}")
    print(f"- {package_dir / 'light_model'}")
    print(f"- {package_dir / 'hard_model_2nd'}")
    print(f"- {package_dir / 'mlp_model'}")
    print(f"- {package_dir / 'package_manifest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
