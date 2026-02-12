#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate hard model stage-1 bbox quality and IoU (x/y/z axes)."""

import os
import sys
import json
import datetime
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

current_dir = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 공통 유틸: 최신 model run / 최신 hard_train_data run 탐색
# ============================================================================

def get_latest_dir_with_checkpoints(base: Path) -> Path:
    """base(10. hard_models_1st)에서 best_x.keras가 있는 최신 run을 반환한다."""
    if not base.exists():
        return None
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        ckpt_x = d / "checkpoints" / "best_x.keras"
        if ckpt_x.exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def get_latest_hard_train_dir(base: Path) -> Path:
    """base(9. hard_train_data)에서 test NPY가 있는 최신 run을 반환한다."""
    if base is None or not base.exists():
        return None
    try:
        subs = [d for d in base.iterdir() if d.is_dir()]
    except OSError:
        return None
    valid = [
        d
        for d in subs
        if (d / "test" / "break_imgs_test.npy").exists()
        and (d / "test" / "break_labels_test.npy").exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda d: d.name)


# ============================================================================
# 라벨 구조 유틸 (9. set_hard_train_data.py / 10. make_hard_model_1st.py와 동일)
# ============================================================================

def slice_roi_targets(y: np.ndarray, roi_idx: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: (N, 1+15K) 반환: (y_cls, y_reg_roi)
    - y_cls: (N,1)   (0/1, break 여부)
    - y_reg_roi: (N, 5K) = [bbox_r(4K), mask_r(K)]
    """
    bbox_dim = 12 * K  # 3*K*4
    mask_dim = 3 * K
    bbox_flat = y[:, 1 : 1 + bbox_dim].astype("float32")
    mask_flat = y[:, 1 + bbox_dim : 1 + bbox_dim + mask_dim].astype("float32")

    bbox = bbox_flat.reshape(-1, 3, K, 4)
    mask = mask_flat.reshape(-1, 3, K)

    bbox_r = bbox[:, roi_idx, :, :].reshape(-1, K * 4)
    mask_r = mask[:, roi_idx, :]

    y_reg_roi = np.concatenate([bbox_r, mask_r], axis=1).astype("float32")
    return y[:, 0:1].astype("float32"), y_reg_roi


def to_corners_np(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """규[hc, hw, dc, dw] (hmin, hmax, dmin, dmax)"""
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    pred: (N, P, 4), gt: (N, K, 4)   IoU: (N, P, K)
    (10. make_hard_model_1st.py iou_matrix_np  일 구조)
    """
    phmin, phmax, pdmin, pdmax = to_corners_np(pred[:, :, None, :])
    thmin, thmax, tdmin, tdmax = to_corners_np(gt[:, None, :, :])

    ihmin = np.maximum(phmin, thmin)
    ihmax = np.minimum(phmax, thmax)
    idmin = np.maximum(pdmin, tdmin)
    idmax = np.minimum(pdmax, tdmax)

    inter_h = np.maximum(0.0, ihmax - ihmin)
    inter_d = np.maximum(0.0, idmax - idmin)
    inter = inter_h * inter_d

    area_p = np.maximum(0.0, phmax - phmin) * np.maximum(0.0, pdmax - pdmin)
    area_t = np.maximum(0.0, thmax - thmin) * np.maximum(0.0, tdmax - tdmin)
    union = area_p + area_t - inter + eps
    return inter / union


# ============================================================================
# 메인  로직
# ============================================================================

def evaluate_axis(model: keras.Model, X: np.ndarray, y: np.ndarray, axis_name: str, roi_idx: int, K: int, P: int, out_dir: Path) -> Dict:
    """
    x/y/z)
      - Best IoU(max PxK) 계산
      - IoU.3/0.5/0.7 비율
      - IoU 스그     """
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    gt_bbox = y_reg[:, : 4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K : 5 * K].reshape(-1, K)
    has_gt = gt_mask.sum(axis=1) > 0

    # 모델 측: (N, 5*P) bbox only (N, P, 4)
    print(f"[{axis_name}] 측 .. (samples={len(X)})")
    pred = model.predict(X, batch_size=32, verbose=0)
    pred_full = pred.reshape(-1, P, 5)
    pred_bbox = pred_full[:, :, :4]

    # IoU (N, P, K) GT 는 분만 효
    iou = iou_matrix_np(pred_bbox, gt_bbox)
    iou_masked = np.where(gt_mask[:, None, :] > 0.5, iou, -1e9)
    flat = iou_masked.reshape(-1, P * K)
    arg = flat.argmax(axis=1)
    best_iou = flat[np.arange(len(flat)), arg]

    best_iou_valid = best_iou[has_gt]
    mean_best_iou = float(best_iou_valid.mean()) if best_iou_valid.size else float("nan")
    med_best_iou = float(np.median(best_iou_valid)) if best_iou_valid.size else float("nan")

    ratio_ge_03 = float((best_iou_valid >= 0.3).mean()) if best_iou_valid.size else 0.0
    ratio_ge_05 = float((best_iou_valid >= 0.5).mean()) if best_iou_valid.size else 0.0
    ratio_ge_07 = float((best_iou_valid >= 0.7).mean()) if best_iou_valid.size else 0.0

    print(f"[{axis_name}] mean Best-IoU={mean_best_iou:.4f}, median={med_best_iou:.4f}, N(valid)={has_gt.sum()}/{len(has_gt)}")
    print(f"[{axis_name}] IoU.3: {ratio_ge_03*100:.1f}%, IoU.5: {ratio_ge_05*100:.1f}%, IoU.7: {ratio_ge_07*100:.1f}%")

    # per-box IoU 분포 (Pred box index별로 IoU 분포 계)
    per_box_stats = []
    all_box_ious = []
    for box_idx in range(P):
        iou_box = iou[:, box_idx, :]  # (N, K)
        iou_box_masked = np.where(gt_mask > 0.5, iou_box, -1e9)
        best_iou_box = iou_box_masked.max(axis=1)  # (N,)
        valid_box = best_iou_box > -0.5e9
        vals = best_iou_box[valid_box]
        if vals.size == 0:
            stats = {
                "box_index": box_idx,
                "mean_iou": None,
                "median_iou": None,
                "ratio_iou_ge_0_3": 0.0,
                "ratio_iou_ge_0_5": 0.0,
                "ratio_iou_ge_0_7": 0.0,
                "num_samples_with_gt": int(valid_box.sum()),
            }
        else:
            stats = {
                "box_index": box_idx,
                "mean_iou": float(vals.mean()),
                "median_iou": float(np.median(vals)),
                "ratio_iou_ge_0_3": float((vals >= 0.3).mean()),
                "ratio_iou_ge_0_5": float((vals >= 0.5).mean()),
                "ratio_iou_ge_0_7": float((vals >= 0.7).mean()),
                "num_samples_with_gt": int(valid_box.sum()),
            }
            all_box_ious.append(vals)
        per_box_stats.append(stats)

    if all_box_ious:
        all_box_ious_flat = np.concatenate(all_box_ious)
        print(f"[{axis_name}] per-box IoU mean={all_box_ious_flat.mean():.4f}, median={np.median(all_box_ious_flat):.4f}")

    # per-box IoU ׷ (box / ü)
    if all_box_ious:
        # (a) ڽ ׷
        fig, axes = plt.subplots(1, P, figsize=(4 * P, 4))
        if P == 1:
            axes = [axes]
        for box_idx, ax in enumerate(axes):
            vals = per_box_stats[box_idx]
            if vals["num_samples_with_gt"] and vals["mean_iou"] is not None:
                v = all_box_ious[box_idx]
                ax.hist(v, bins=30, alpha=0.8, edgecolor="white")
                ax.set_title(f"Box {box_idx} IoU")
            ax.set_xlabel("IoU")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        per_box_path = out_dir / f"iou_hist_per_box_{axis_name}.png"
        plt.savefig(per_box_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[{axis_name}] ", per_box_path)

        # (b) ü ڽ  ׷
        fig, ax_all = plt.subplots(figsize=(6, 4))
        ax_all.hist(all_box_ious_flat, bins=40, color="C2", alpha=0.8, edgecolor="white")
        ax_all.set_xlabel("IoU")
        ax_all.set_ylabel("Count")
        ax_all.set_title(f"[{axis_name.upper()}] IoU distribution over all predicted boxes")
        ax_all.grid(True, alpha=0.3)
        plt.tight_layout()
        per_box_all_path = out_dir / f"iou_hist_per_box_all_{axis_name}.png"
        plt.savefig(per_box_all_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[{axis_name}] ", per_box_all_path)

    # Best-pair 기 GT/Pred bbox 추출 (arg서 GT/Pred 덱복원)
    pred_idx = arg // K
    gt_idx = arg % K
    pred_sel = pred_bbox[np.arange(len(pred_bbox)), pred_idx]  # (N,4)
    gt_sel = gt_bbox[np.arange(len(gt_bbox)), gt_idx]          # (N,4)

    # has_gt=True 플용
    pred_sel_valid = pred_sel[has_gt]
    gt_sel_valid = gt_sel[has_gt]

    # height / degree center 차 계수
    rmse = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    corr_hc = np.nan
    corr_dc = np.nan
    if gt_sel_valid.size:
        err = pred_sel_valid - gt_sel_valid
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        hc_true = gt_sel_valid[:, 0]
        hc_pred = pred_sel_valid[:, 0]
        dc_true = gt_sel_valid[:, 2]
        dc_pred = pred_sel_valid[:, 2]
        if np.std(hc_true) > 1e-8 and np.std(hc_pred) > 1e-8:
            corr_hc = float(np.corrcoef(hc_true, hc_pred)[0, 1])
        if np.std(dc_true) > 1e-8 and np.std(dc_pred) > 1e-8:
            corr_dc = float(np.corrcoef(dc_true, dc_pred)[0, 1])

    print(f"[{axis_name}] RMSE (hc, hw, dc, dw) =", [float(x) for x in rmse])
    print(f"[{axis_name}] Correlation hc={corr_hc:.3f} dc={corr_dc:.3f}" if not np.isnan(corr_hc) and not np.isnan(corr_dc) else f"[{axis_name}] Correlation hc={corr_hc} dc={corr_dc}")

    # (1) IoU ׷
    fig, ax = plt.subplots(figsize=(6, 4))
    if best_iou_valid.size:
        ax.hist(best_iou_valid, bins=30, color="C0", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Best IoU (max over P×K)")
    ax.set_ylabel("Count")
    ax.set_title(f"[{axis_name.upper()}] Best IoU Distribution (Test)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"iou_hist_{axis_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{axis_name}]  {out_path}")

    # (2) GT vs Pred center scatter (hc, dc)
    if gt_sel_valid.size:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # height center
        axes[0].scatter(hc_true, hc_pred, s=8, alpha=0.5, color="C0")
        min_h = float(min(hc_true.min(), hc_pred.min()))
        max_h = float(max(hc_true.max(), hc_pred.max()))
        axes[0].plot([min_h, max_h], [min_h, max_h], "k--", lw=1)
        axes[0].set_xlabel("GT hc")
        axes[0].set_ylabel("Pred hc")
        axes[0].set_title(f"[{axis_name.upper()}] Height center (hc)\nRMSE={rmse[0]:.3f}, r={corr_hc:.3f}")
        axes[0].grid(True, alpha=0.3)

        # degree center
        axes[1].scatter(dc_true, dc_pred, s=8, alpha=0.5, color="C1")
        min_d = float(min(dc_true.min(), dc_pred.min()))
        max_d = float(max(dc_true.max(), dc_pred.max()))
        axes[1].plot([min_d, max_d], [min_d, max_d], "k--", lw=1)
        axes[1].set_xlabel("GT dc")
        axes[1].set_ylabel("Pred dc")
        axes[1].set_title(f"[{axis_name.upper()}] Degree center (dc)\nRMSE={rmse[2]:.3f}, r={corr_dc:.3f}")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = out_dir / f"center_scatter_{axis_name}.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[{axis_name}] ", scatter_path)

        # (3) IoU vs center error (radius) scatter
        center_err = np.sqrt((hc_pred - hc_true) ** 2 + (dc_pred - dc_true) ** 2)
        fig, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(center_err, best_iou_valid, s=8, alpha=0.5, color="C2")
        ax2.set_xlabel("Center error (sqrt((hc)^2 + (dc)^2))")
        ax2.set_ylabel("Best IoU")
        ax2.set_title(f"[{axis_name.upper()}] IoU vs center error")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        iou_err_path = out_dir / f"iou_vs_center_error_{axis_name}.png"
        plt.savefig(iou_err_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[{axis_name}] ", iou_err_path)

    return {
        "axis": axis_name,
        "roi_idx": roi_idx,
        "mean_best_iou": mean_best_iou,
        "median_best_iou": med_best_iou,
        "num_samples_with_gt": int(has_gt.sum()),
        "num_samples_total": int(len(has_gt)),
        "ratio_iou_ge_0_3": ratio_ge_03,
        "ratio_iou_ge_0_5": ratio_ge_05,
        "ratio_iou_ge_0_7": ratio_ge_07,
        "rmse_hc": float(rmse[0]),
        "rmse_hw": float(rmse[1]),
        "rmse_dc": float(rmse[2]),
        "rmse_dw": float(rmse[3]),
        "corr_hc": float(corr_hc) if not np.isnan(corr_hc) else None,
        "corr_dc": float(corr_dc) if not np.isnan(corr_dc) else None,
        "per_box_stats": per_box_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hard 1차 모델 bbox 품질/IoU 평가"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="평가할 run 디렉터리 (미지정 시 최신 run 자동 선택)",
    )
    parser.add_argument(
        "--target-mean-best-iou",
        type=float,
        default=0.45,
        help="재학습 기준: mean_best_iou",
    )
    parser.add_argument(
        "--target-ratio-iou-0-5",
        type=float,
        default=0.55,
        help="재학습 기준: IoU>=0.5 비율",
    )
    parser.add_argument(
        "--target-ratio-iou-0-7",
        type=float,
        default=0.30,
        help="재학습 기준: IoU>=0.7 비율",
    )
    parser.add_argument(
        "--target-pass-mode",
        type=str,
        choices=["all_axes", "average"],
        default="average",
        help="pass 판정 방식",
    )
    args = parser.parse_args()

    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # 1) Resolve model run and evaluation data run
    models_base = Path(current_dir) / "10. hard_models_1st"
    hard_data_base = Path(current_dir) / "9. hard_train_data"

    if args.run_dir is not None:
        # User-selected run directory
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run_dir does not exist: {run_dir}")
        if not (run_dir / "checkpoints" / "best_x.keras").exists():
            raise FileNotFoundError(
                f"checkpoints/best_x.keras not found under --run_dir: {run_dir}"
            )
    else:
        # Default: latest valid run under 10. hard_models_1st
        run_dir = get_latest_dir_with_checkpoints(models_base)
        if run_dir is None:
            raise FileNotFoundError(
                f"10. hard_models_1st에서 checkpoints/best_x.keras가 있는 run을 찾지 못했습니다: {models_base}"
            )

    run_name = run_dir.name

    data_run_dir = get_latest_hard_train_dir(hard_data_base)
    if data_run_dir is None:
        raise FileNotFoundError(
            f"9. hard_train_data에서 test/break_imgs_test.npy가 있는 run을 찾지 못했습니다: {hard_data_base}"
        )

    print(f"모델 run: {run_name} ({run_dir})")
    print(f"평가 데이터 run: {data_run_dir.name} ({data_run_dir})")

    test_dir = data_run_dir / "test"
    X_test_path = test_dir / "break_imgs_test.npy"
    y_test_path = test_dir / "break_labels_test.npy"
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"테스트 NPY를 찾지 못했습니다: {X_test_path}, {y_test_path}")

    X_test = np.load(X_test_path).astype(np.float32)
    y_test = np.load(y_test_path).astype(np.float32)
    K = int((y_test.shape[1] - 1) // 15)
    assert 1 + 15 * K == y_test.shape[1], f"y_test.shape[1]={y_test.shape[1]} not 1+15*K"
    print("X_test:", X_test.shape, "y_test:", y_test.shape, "K:", K)

    # 2)  ε (ະ best_x/y/z)
    ckpt_dir = run_dir / "checkpoints"
    best_x_path = ckpt_dir / "best_x.keras"
    best_y_path = ckpt_dir / "best_y.keras"
    best_z_path = ckpt_dir / "best_z.keras"
    for p in [best_x_path, best_y_path, best_z_path]:
        if not p.exists():
            raise FileNotFoundError(f"체크인 찾을 습다: {p}")

    print("모델 로드 ..")
    model_x = keras.models.load_model(str(best_x_path), compile=False)
    model_y = keras.models.load_model(str(best_y_path), compile=False)
    model_z = keras.models.load_model(str(best_z_path), compile=False)
    print("모델 로드 료.")

    # 출력 렉리: 11. evaluate_hard_model_1st/<각>/
    eval_base = Path(current_dir) / "11. evaluate_hard_model_1st"
    eval_dir = eval_base / run_name
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(" 결과 렉리:", eval_dir)

    # Dense 출력 차원로P 계산 (5*P)
    out_dim = model_x.output_shape[-1]
    if out_dim % 5 != 0:
        raise ValueError(f"출력 차원 {out_dim} ) 5*P 태 닙다.")
    P = out_dim // 5
    print(f"측 박스 개수 P: {P}")

    # 3) 축별 
    axis_infos = [
        ("x", 0, model_x),
        ("y", 1, model_y),
        ("z", 2, model_z),
    ]
    axis_metrics = []
    for axis_name, roi_idx, m in axis_infos:
        axis_metrics.append(
            evaluate_axis(m, X_test, y_test, axis_name, roi_idx=roi_idx, K=K, P=P, out_dir=eval_dir)
        )

    # 3-1) x/y/z 개별 나친 약  성
    def _combine_pngs(prefix: str, title: str):
        """center_scatter_x/y/z.png 같 31개로 친"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, axis_name in zip(axes, ["x", "y", "z"]):
            img_path = eval_dir / f"{prefix}_{axis_name}.png"
            if img_path.exists():
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(axis_name.upper())
            ax.axis("off")
        fig.suptitle(title)
        plt.tight_layout()
        out_path = eval_dir / f"{prefix}_all_axes.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("(3친 ):", out_path)

    _combine_pngs("center_scatter", "GT vs Pred center (hc, dc) by axis")
    _combine_pngs("iou_hist", "Best IoU distribution by axis")
    _combine_pngs("iou_vs_center_error", "IoU vs center error by axis")

    # 3-2) Plot training curves from histories.json
    histories_path = run_dir / "histories.json"
    if histories_path.exists():
        try:
            with open(histories_path, "r", encoding="utf-8") as f:
                histories = json.load(f)
            # 축별loss / val_bbox_iou 곡선 롯
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.reshape(2, 3)
            for idx, axis_name in enumerate(["x", "y", "z"]):
                h = histories.get(axis_name)
                if not h:
                    continue
                # loss
                ax_loss = axes[0, idx]
                loss = h.get("loss", [])
                val_loss = h.get("val_loss", [])
                ax_loss.plot(loss, label="loss")
                if val_loss:
                    ax_loss.plot(val_loss, label="val_loss")
                ax_loss.set_title(f"{axis_name.upper()} loss")
                ax_loss.set_xlabel("epoch")
                ax_loss.set_ylabel("loss")
                ax_loss.grid(True, alpha=0.3)
                ax_loss.legend()
                # bbox_iou
                ax_iou = axes[1, idx]
                iou = h.get("bbox_iou", [])
                val_iou = h.get("val_bbox_iou", [])
                ax_iou.plot(iou, label="bbox_iou")
                if val_iou:
                    ax_iou.plot(val_iou, label="val_bbox_iou")
                ax_iou.set_title(f"{axis_name.upper()} bbox_iou")
                ax_iou.set_xlabel("epoch")
                ax_iou.set_ylabel("IoU")
                ax_iou.grid(True, alpha=0.3)
                ax_iou.legend()
            fig.suptitle("Training curves (loss / IoU) per axis", y=1.02)
            plt.tight_layout()
            curves_path = eval_dir / "training_curves.png"
            plt.savefig(curves_path, dpi=150, bbox_inches="tight")
            plt.close()
            print("", curves_path)
        except Exception as e:
            print("습 곡선 롯 성 류:", e)

    # 4) 체 약  (로트 목적 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) Per-axis mean IoU bar plot
    names = [m["axis"].upper() for m in axis_metrics]
    means = [m["mean_best_iou"] for m in axis_metrics]
    axes[0, 0].bar(names, means, color=["C0", "C1", "C2"])
    for x, v in zip(names, means):
        axes[0, 0].text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Mean Best IoU")
    axes[0, 0].set_title("Per-axis break location accuracy (Mean Best IoU, Test)")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    # (2) IoU>=0.5  ׷
    ratio_ge_05 = [m["ratio_iou_ge_0_5"] for m in axis_metrics]
    axes[0, 1].bar(names, ratio_ge_05, color=["C0", "C1", "C2"])
    for x, v in zip(names, ratio_ge_05):
        axes[0, 1].text(x, v + 0.01, f"{v*100:.1f}%", ha="center", va="bottom")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Ratio IoU 0.5")
    axes[0, 1].set_title("Per-axis ratio of IoU 0.5")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # (3) IoU>=0.7  ׷
    ratio_ge_07 = [m["ratio_iou_ge_0_7"] for m in axis_metrics]
    axes[1, 0].bar(names, ratio_ge_07, color=["C0", "C1", "C2"])
    for x, v in zip(names, ratio_ge_07):
        axes[1, 0].text(x, v + 0.01, f"{v*100:.1f}%", ha="center", va="bottom")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Ratio IoU 0.7")
    axes[1, 0].set_title("Per-axis ratio of IoU 0.7")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # (4) Text overview link to project goal
    axes[1, 1].axis("off")
    total_with_gt = sum(m["num_samples_with_gt"] for m in axis_metrics)
    overview_lines = [
        "SMARTCS Pole Hard model (break location bbox) evaluation summary",
        "",
        f"Model run: {run_name}",
        f"Eval data: 9. hard_train_data/{data_run_dir.name}/test (break_imgs_test.npy)",
        "",
        "Per-axis Mean Best IoU:",
        "  " + ", ".join([f"{m['axis'].upper()}={m['mean_best_iou']:.3f}" for m in axis_metrics]),
        "",
        "Per-axis ratio IoU 0.5:",
        "  " + ", ".join([f"{m['axis'].upper()}={m['ratio_iou_ge_0_5']*100:.1f}%" for m in axis_metrics]),
        "Per-axis ratio IoU 0.7:",
        "  " + ", ".join([f"{m['axis'].upper()}={m['ratio_iou_ge_0_7']*100:.1f}%" for m in axis_metrics]),
        "",
        f"Samples with GT break bbox (sum over axes): {total_with_gt}",
        "",
        "These metrics show how well the hard model localizes break regions",
        "   on the pole surface using bounding boxes.",
        "   Higher IoU 0.5 / 0.7 ratios mean more accurate break localization,",
        "   which directly supports the project goal of reliable break detection.",
    ]
    axes[1, 1].text(
        0.0, 1.0, "\n".join(overview_lines),
        ha="left", va="top", fontsize=9,
    )

    plt.tight_layout()
    overview_path = eval_dir / "overview_summary.png"
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("", overview_path)

    # 5) Save metrics as JSON
    metrics = {
        "model_run": run_name,
        "data_run": data_run_dir.name,
        "evaluation_time": datetime.datetime.now().isoformat(),
        "num_samples_test": int(len(X_test)),
        "K": int(K),
        "P": int(P),
        "axes": axis_metrics,
    }
    metrics_path = eval_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("", metrics_path)

    mean_iou_avg = float(np.mean([m["mean_best_iou"] for m in axis_metrics]))
    ratio05_avg = float(np.mean([m["ratio_iou_ge_0_5"] for m in axis_metrics]))
    ratio07_avg = float(np.mean([m["ratio_iou_ge_0_7"] for m in axis_metrics]))

    axis_pass = {}
    for m in axis_metrics:
        axis_name = m["axis"]
        axis_pass[axis_name] = (
            (m["mean_best_iou"] >= args.target_mean_best_iou)
            and (m["ratio_iou_ge_0_5"] >= args.target_ratio_iou_0_5)
            and (m["ratio_iou_ge_0_7"] >= args.target_ratio_iou_0_7)
        )

    if args.target_pass_mode == "all_axes":
        overall_pass = all(axis_pass.values())
    else:
        overall_pass = (
            (mean_iou_avg >= args.target_mean_best_iou)
            and (ratio05_avg >= args.target_ratio_iou_0_5)
            and (ratio07_avg >= args.target_ratio_iou_0_7)
        )

    weak_axes = sorted(
        axis_metrics,
        key=lambda m: (m["mean_best_iou"], m["ratio_iou_ge_0_5"], m["ratio_iou_ge_0_7"]),
    )
    feedback = {
        "stage": "hard_model_1st",
        "evaluation_dir": str(eval_dir),
        "model_run": run_name,
        "data_run": data_run_dir.name,
        "pass": bool(overall_pass),
        "recommended_retrain": bool(not overall_pass),
        "pass_mode": args.target_pass_mode,
        "criteria": {
            "target_mean_best_iou": float(args.target_mean_best_iou),
            "target_ratio_iou_0_5": float(args.target_ratio_iou_0_5),
            "target_ratio_iou_0_7": float(args.target_ratio_iou_0_7),
        },
        "actual": {
            "avg_mean_best_iou": mean_iou_avg,
            "avg_ratio_iou_0_5": ratio05_avg,
            "avg_ratio_iou_0_7": ratio07_avg,
        },
        "axis_pass": axis_pass,
        "weak_axes_order": [m["axis"] for m in weak_axes],
    }
    feedback_path = eval_dir / "training_feedback.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print("", feedback_path)
    print(
        "[feedback] pass={} mode={} avg(mean_iou={:.4f}, iou>=0.5={:.4f}, iou>=0.7={:.4f})".format(
            overall_pass, args.target_pass_mode, mean_iou_avg, ratio05_avg, ratio07_avg
        )
    )

    # 6)  명나txt 일성
    desc_lines = [
        "Image descriptions for hard model (first-stage bbox evaluation):",
        "",
        "iou_hist_x/y/z.png: Best IoU distribution (per-sample best predicted box vs GT) for each axis.",
        "center_scatter_x/y/z.png: GT vs Pred height/degree centers (hc, dc) for best IoU pair, per axis.",
        "iou_vs_center_error_x/y/z.png: Relationship between center error and Best IoU, per axis.",
        "iou_hist_per_box_x/y/z.png: IoU distribution for each predicted box index (P boxes), per axis.",
        "iou_hist_per_box_all_x/y/z.png: IoU distribution over all predicted boxes (flattened), per axis.",
        "center_scatter_all_axes.png: GT vs Pred centers (hc, dc) for X/Y/Z axes side by side.",
        "iou_hist_all_axes.png: Best IoU histograms for X/Y/Z axes in one figure.",
        "iou_vs_center_error_all_axes.png: IoU vs center error plots for X/Y/Z axes in one figure.",
        "training_curves.png: Training/validation loss and bbox_iou curves for X/Y/Z hard models.",
        "overview_summary.png: Summary of axis-wise mean IoU, IoU.5/0.7 ratios, and evaluation context.",
    ]
    desc_path = eval_dir / "image_descriptions.txt"
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(desc_lines))
    print("명 ", desc_path)


if __name__ == "__main__":
    main()
