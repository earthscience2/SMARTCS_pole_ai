#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard 모델 2차 평가 - bbox + confidence 예측 성능 평가 및 파단 위치 시각화"""

import os
import sys
import subprocess
from pathlib import Path

# Windows에서 --local 없이 실행 시 WSL2 스크립트로 넘겨서 GPU 평가
_run_local = "--local" in sys.argv or sys.platform != "win32"
if _run_local:
    if "--local" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--local"]
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parent
    # 2차 Hard 모델 평가 전용 WSL 스크립트
    _sh_candidates = [
        Path(_script_dir) / "13. evaluate_hard_model_2nd_wsl2.sh",
        Path(_script_dir) / "evaluate_hard_model_2nd_wsl2.sh",
        _project_root / "make_ai" / "13. evaluate_hard_model_2nd_wsl2.sh",
        _project_root / "make_ai" / "evaluate_hard_model_2nd_wsl2.sh",
    ]
    _sh_path = next((p for p in _sh_candidates if p.exists()), None)
    if _sh_path is not None:
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = (
            "/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")
        ) if _drive else str(_abs).replace("\\", "/")
        print("WSL2에서 GPU 평가(2nd hard model) 실행:", _wsl_path)
        ret = subprocess.run(
            ["wsl", "bash", _wsl_path] + sys.argv[1:],
            cwd=str(_project_root),
        )
        sys.exit(ret.returncode)
    # 스크립트 없으면 로컬 실행 진행
import json
import pickle
import re
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

# 그래프 라벨은 영어 사용 (폰트 미설치 환경에서도 깨지지 않음)
matplotlib.rcParams["axes.unicode_minus"] = False

import tensorflow as tf
from tensorflow import keras

# GPU 사용 설정: 메모리 증가 방식으로 할당해 OOM 방지
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 사용:", [gpu.name for gpu in gpus])
    except RuntimeError as e:
        print("GPU 설정 실패:", e)
else:
    print("GPU 없음, CPU 사용")

current_dir = os.path.dirname(os.path.abspath(__file__))
AXIS_NAMES = ["x", "y", "z"]
IMG_HEIGHT = 304


# 2nd-stage hard model (confidence head) runs (12. hard_models_2nd) + 1st-stage bbox runs (10. hard_models_1st)

SECOND_STAGE_BASE = Path(current_dir) / "12. hard_models_2nd"
FIRST_STAGE_BASE = Path(current_dir) / "10. hard_models_1st"


def _get_latest_second_stage_run(base: Path) -> Optional[Path]:
    """12. hard_models_2nd 아래에서 최신 run 디렉터리(=2차 hard 모델)를 반환."""
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.name)

# =============================================================================
# 전처리 유틸 (6. training_data_bbox.py 기능을 내장, 외부 파일 의존 제거)
# =============================================================================

def load_crop_csv(csv_path: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """크롭된 CSV 파일 로드."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        if verbose:
            print("❌ read_csv failed:", csv_path)
            print("   err =", repr(e))
        return None


def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = "height",
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    max_height: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """CSV에서 시퀀스(이미지) 생성 (0~1 정규화)."""
    df = load_crop_csv(csv_path)
    if df is None:
        return None
    required_cols = ["height", "degree", "x_value", "y_value", "z_value"]
    if any(c not in df.columns for c in required_cols):
        return None
    if sort_by == "height":
        df = df.sort_values(["height", "degree"]).reset_index(drop=True)
    else:
        df = df.sort_values(["degree", "height"]).reset_index(drop=True)
    if feature_min_max is None:
        feature_min_max = {
            "height": (df["height"].min(), df["height"].max()),
            "degree": (df["degree"].min(), df["degree"].max()),
            "x_value": (df["x_value"].min(), df["x_value"].max()),
            "y_value": (df["y_value"].min(), df["y_value"].max()),
            "z_value": (df["z_value"].min(), df["z_value"].max()),
        }
    for col in ["x_value", "y_value", "z_value"]:
        vmin, vmax = feature_min_max[col]
        df[col] = (df[col].astype(np.float32) - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    heights = np.sort(df["height"].unique())
    degrees = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)
    H, W = len(heights), len(degrees)
    df["degree"] = (np.round(df["degree"] / 5.0) * 5.0).astype(np.float32)
    dmin, dmax = df["degree"].min(), df["degree"].max()
    if dmax <= 90.0:
        df["degree"] += 90.0
    elif dmin >= 180.0 and dmax <= 270.0:
        df["degree"] -= 90.0
    elif dmin >= 270.0:
        df["degree"] -= 180.0

    def _make_grid(col: str) -> np.ndarray:
        g = df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean").reindex(index=heights, columns=degrees).to_numpy(dtype=np.float32)
        return np.nan_to_num(g, nan=0.0)
    img = np.stack([_make_grid("x_value"), _make_grid("y_value"), _make_grid("z_value")], axis=-1).astype(np.float32)
    metadata = {
        "grid_shape": img.shape,
        "num_points": int(len(df)),
        "original_length": int(len(df)),
        "unique_heights": int(H),
        "unique_degrees": int(W),
        "height_values": heights.tolist(),
        "degree_values": degrees.tolist(),
        "feature_min_max": {k: list(v) for k, v in feature_min_max.items()},
    }
    return img, metadata


def collect_all_crop_files(data_dir: str, is_break: bool) -> List[Tuple[str, str, str, int]]:
    """data_dir/{break|normal} 아래 *_OUT_processed.csv 수집. 반환: (csv_path, project_name, poleid, label)."""
    base_dir = Path(current_dir) / data_dir / ("break" if is_break else "normal")
    out = []
    if not base_dir.exists():
        return out
    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                out.append((str(csv_file), project_dir.name, pole_dir.name, 1 if is_break else 0))
    return out


def _get_sample_id_from_csv(csv_path: str) -> Optional[str]:
    """0621R481_2_OUT_processed.csv -> 0621R481_2"""
    m = re.match(r"(.+)_OUT_processed\.csv$", Path(csv_path).name)
    return m.group(1) if m else None


def match_roi_json_from_csv(csv_path: str, edit_data_dir: str = "5. edit_data") -> Optional[str]:
    """CSV에 대응하는 roi_info.json 경로. 5. edit_data 없으면 4. merge_data에서도 찾음."""
    p = Path(csv_path)
    sample_id = _get_sample_id_from_csv(csv_path)
    if sample_id is None or len(p.parents) < 3:
        return None
    project, poleid = p.parents[1].name, p.parents[0].name
    for try_dir in (edit_data_dir, "4. merge_data"):
        roi_json = Path(current_dir) / try_dir / "break" / project / poleid / f"{sample_id}_OUT_processed_roi_info.json"
        if roi_json.exists():
            return str(roi_json)
    return None


def load_roi_info_json(json_path: str) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_roi_bbox(roi_info: dict, k: int) -> List[List[float]]:
    """roi_{k}_regions에서 [hc, hw, dc, dw] 리스트."""
    out = []
    for r in (roi_info.get(f"roi_{k}_regions") or []):
        if not isinstance(r, dict):
            continue
        hmin, hmax = r.get("height_min"), r.get("height_max")
        dmin, dmax = r.get("degree_min"), r.get("degree_max")
        if None in (hmin, hmax, dmin, dmax):
            continue
        try:
            hmin, hmax, dmin, dmax = map(float, (hmin, hmax, dmin, dmax))
        except Exception:
            continue
        out.append([(hmin + hmax) / 2.0, hmax - hmin, (dmin + dmax) / 2.0, dmax - dmin])
    return out


def expand_rois_from_roi_info(roi_info: Optional[dict]) -> List[Tuple[int, List[float]]]:
    """(roi_idx, [hc, hw, dc, dw]) 리스트."""
    if roi_info is None:
        return []
    return [(k, b) for k in (0, 1, 2) for b in _parse_roi_bbox(roi_info, k)]


def resize_img_height(img: np.ndarray, target_h: int = IMG_HEIGHT) -> np.ndarray:
    h, w, c = img.shape
    if h == target_h:
        return img
    return zoom(img, (target_h / h, 1.0, 1.0), order=1).astype(np.float32)


# =============================================================================

def to_corners(box: np.ndarray):
    hc, hw, dc, dw = [box[..., i] for i in range(4)]
    return hc - 0.5 * hw, hc + 0.5 * hw, dc - 0.5 * dw, dc + 0.5 * dw


def iou_matrix(pred: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    phmin, phmax, pdmin, pdmax = to_corners(pred[:, None, :])
    thmin, thmax, tdmin, tdmax = to_corners(pred[None, :, :])
    ihmin, ihmax = np.maximum(phmin, thmin), np.minimum(phmax, thmax)
    idmin, idmax = np.maximum(pdmin, tdmin), np.minimum(pdmax, tdmax)
    inter = np.maximum(0.0, ihmax - ihmin) * np.maximum(0.0, idmax - idmin)
    area_p = np.maximum(0.0, phmax - phmin) * np.maximum(0.0, pdmax - pdmin)
    area_t = np.maximum(0.0, thmax - thmin) * np.maximum(0.0, tdmax - tdmin)
    return inter / (area_p + area_t - inter + eps)


def select_consistent_box(pred_boxes: np.ndarray) -> Tuple[int, float]:
    """후보 박스 중 일관성(평균 IoU)이 가장 높은 것 선택 (4*P 모델용)."""
    if pred_boxes.ndim != 2 or pred_boxes.shape[0] <= 1:
        return 0, 0.0
    iou_mat = iou_matrix(pred_boxes)
    iou_sum = (iou_mat.sum(axis=1) - np.diag(iou_mat)) / (pred_boxes.shape[0] - 1)
    best_idx = int(np.argmax(iou_sum))
    return best_idx, float(iou_sum[best_idx])


def box_iou_physical(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float], eps: float = 1e-7) -> float:
    """물리 좌표 (dmin, dmax, hmin, hmax) 두 박스의 IoU."""
    da_min, da_max, ha_min, ha_max = box_a
    db_min, db_max, hb_min, hb_max = box_b
    di_min = max(da_min, db_min)
    di_max = min(da_max, db_max)
    hi_min = max(ha_min, hb_min)
    hi_max = min(ha_max, hb_max)
    inter = max(0.0, di_max - di_min) * max(0.0, hi_max - hi_min)
    area_a = max(0.0, da_max - da_min) * max(0.0, ha_max - ha_min)
    area_b = max(0.0, db_max - db_min) * max(0.0, hb_max - hb_min)
    union = area_a + area_b - inter
    return inter / (union + eps)


def decode_bbox_norm_to_physical(box: np.ndarray, feature_min_max: Dict) -> Tuple[float, float, float, float]:
    h_min, h_max = feature_min_max["height"]
    d_min, d_max = feature_min_max["degree"]
    h_span = max(h_max - h_min, 1e-6)
    d_span = max(d_max - d_min, 1e-6)
    hc, hw, dc, dw = [float(np.clip(box[i], 0, 1)) for i in range(4)]
    h_center = h_min + hc * h_span
    d_center = d_min + dc * d_span
    h_width = min(max(hw, 0), 1) * h_span
    d_width = min(max(dw, 0), 1) * d_span
    hmin = max(h_min, h_center - 0.5 * h_width)
    hmax = min(h_max, h_center + 0.5 * h_width)
    dmin = max(d_min, d_center - 0.5 * d_width)
    dmax = min(d_max, d_center + 0.5 * d_width)
    return float(dmin), float(dmax), float(hmin), float(hmax)


# 9. hard_train_data run의 break_labels.npy 라벨 형식 (9. set_hard_train_data.py와 동일)
_HARD_LABEL_K = 10
_HARD_LABEL_DIM = 1 + (3 * _HARD_LABEL_K * 4) + (3 * _HARD_LABEL_K)


def load_gt_bbox_from_hard_run(run_dir: Path) -> Optional[Dict[str, Dict[str, List[Tuple[float, float, float, float]]]]]:
    """
    9. hard_train_data/<run> 의 break_imgs_metadata.json + break_labels.npy에서
    break 샘플별 GT bbox(물리 좌표)를 로드. csv_path(정규화) -> axis -> [(dmin,dmax,hmin,hmax), ...]
    ROI JSON이 없을 때 IoU 계산용으로 사용.
    """
    meta_path = run_dir / "break_imgs_metadata.json"
    labels_path = run_dir / "break_labels.npy"
    if not meta_path.exists() or not labels_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        samples = meta_dict.get("samples", [])
        y_all = np.load(labels_path)
    except Exception:
        return None
    if len(samples) != y_all.shape[0] or y_all.shape[1] != _HARD_LABEL_DIM:
        return None
    out = {}
    off_bbox = 1
    len_bbox = 3 * _HARD_LABEL_K * 4
    len_mask = 3 * _HARD_LABEL_K
    for i, sample in enumerate(samples):
        y_vec = y_all[i]
        if y_vec[0] < 0.5:
            continue
        fmm = sample.get("feature_min_max")
        if not fmm or "height" not in fmm or "degree" not in fmm:
            continue
        bbox_flat = y_vec[off_bbox : off_bbox + len_bbox].reshape(3, _HARD_LABEL_K, 4)
        mask_flat = y_vec[off_bbox + len_bbox : off_bbox + len_bbox + len_mask].reshape(3, _HARD_LABEL_K)
        gt_by_axis = {}
        for axis_idx, axis_name in enumerate(AXIS_NAMES):
            boxes = []
            for k in range(_HARD_LABEL_K):
                if mask_flat[axis_idx, k] < 0.5:
                    continue
                box_n = bbox_flat[axis_idx, k]
                try:
                    phys = decode_bbox_norm_to_physical(box_n, fmm)
                    boxes.append(phys)
                except Exception:
                    continue
            if boxes:
                gt_by_axis[axis_name] = boxes
        if gt_by_axis:
            key = str(Path(sample.get("csv_path", "")).resolve())
            if key:
                out[key] = gt_by_axis
    return out if out else None


def _make_grid(df, col, heights, degrees):
    g = df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean").reindex(index=heights, columns=degrees).to_numpy(dtype=np.float32)
    return np.nan_to_num(g, nan=0.0)


def _box_passes_size_filter(
    box: Tuple[float, float, float, float],
    data_extent: Optional[Dict],
    min_box_ratio: float,
    min_box_degree_span: float,
    min_box_height_span: float,
) -> bool:
    """조건(비율·절대 크기)을 만족하면 True."""
    dmin, dmax, hmin, hmax = box
    box_deg_span = max(dmax - dmin, 0)
    box_h_span = max(hmax - hmin, 0)
    if min_box_degree_span > 0 and box_deg_span < min_box_degree_span:
        return False
    if min_box_height_span > 0 and box_h_span < min_box_height_span:
        return False
    if min_box_ratio > 0 and data_extent is not None:
        deg_span = max(data_extent["degree_max"] - data_extent["degree_min"], 1e-9)
        h_span = max(data_extent["height_max"] - data_extent["height_min"], 1e-9)
        if (box_deg_span / deg_span) < min_box_ratio or (box_h_span / h_span) < min_box_ratio:
            return False
    return True


# 상위 3개 박스 색: 1위 초록, 2위 노랑, 3위 빨강
RANK_COLORS = ["lime", "yellow", "red"]


def plot_csv_with_boxes(
    csv_path: str,
    boxes_by_axis: Dict[str, List],
    best_box_by_axis: Dict[str, Tuple],
    score_by_axis: Dict[str, float],
    output_file: Path,
    conf_by_axis: Optional[Dict[str, np.ndarray]] = None,
    draw_candidates: bool = False,
    confidence_threshold: float = 0.0,
    use_confidence: bool = False,
    top3_by_axis: Optional[Dict[str, List[Tuple[Tuple[float, float, float, float], float]]]] = None,
    min_box_ratio: float = 0.0,
    min_box_degree_span: float = 5,
    min_box_height_span: float = 0.1,
    highlight_rank_by_axis: Optional[Dict[str, int]] = None,
    top3_iou_by_axis: Optional[Dict[str, List[Optional[float]]]] = None,
    draw_only_highlighted: bool = False,
) -> None:
    df = load_crop_csv(csv_path)
    if df is None or df.empty or any(c not in df.columns for c in ["height", "degree", "x_value", "y_value", "z_value"]):
        return
    heights, degrees = sorted(df["height"].unique()), sorted(df["degree"].unique())
    deg_min, deg_max = min(degrees), max(degrees)
    h_min, h_max = min(heights), max(heights)
    data_deg_span = max(deg_max - deg_min, 1e-9)
    data_h_span = max(h_max - h_min, 1e-9)
    x_grid = _make_grid(df, "x_value", heights, degrees)
    y_grid = _make_grid(df, "y_value", heights, degrees)
    z_grid = _make_grid(df, "z_value", heights, degrees)
    D, H = np.meshgrid(degrees, heights)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (ax, grid, title, axis_name) in enumerate(zip(axes, [x_grid, y_grid, z_grid], ["X Value", "Y Value", "Z Value"], AXIS_NAMES)):
        v = grid[~np.isnan(grid)]
        vmin = np.percentile(v, 2) if len(v) else 0
        vmax = np.percentile(v, 98) if len(v) else 1
        ax.contourf(D, H, grid, levels=30, cmap="RdBu_r", vmin=vmin, vmax=vmax, extend="both")
        ax.contour(D, H, grid, levels=30, colors="black", linewidths=0.5, alpha=0.3)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Height (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        score = score_by_axis.get(axis_name, 0.0)
        if use_confidence and score < confidence_threshold:
            ax.text(0.02, 0.98, f"conf={score:.3f} (<{confidence_threshold})", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="gray")
            continue
        if score >= confidence_threshold or not use_confidence:
            if top3_by_axis and axis_name in top3_by_axis:
                ranked = top3_by_axis[axis_name]
                iou_list = (top3_iou_by_axis or {}).get(axis_name)
                for rank, (box, s) in enumerate(ranked):
                    if draw_only_highlighted and highlight_rank_by_axis is not None:
                        if highlight_rank_by_axis.get(axis_name) != rank:
                            continue
                    if rank >= len(RANK_COLORS):
                        break
                    dmin, dmax, hmin, hmax = box
                    is_criterion = highlight_rank_by_axis and highlight_rank_by_axis.get(axis_name) == rank
                    color = "red" if is_criterion else (RANK_COLORS[rank] if not draw_only_highlighted else "red")
                    ax.add_patch(patches.Rectangle((dmin, hmin), dmax - dmin, hmax - hmin, linewidth=2 if rank == 0 or is_criterion else 1.5, edgecolor=color, facecolor="none", alpha=0.9))
                    score_text = f"conf={s:.3f}" if use_confidence else f"score={s:.3f}"
                    if iou_list and rank < len(iou_list) and iou_list[rank] is not None:
                        score_text += f" iou={iou_list[rank]:.3f}"
                    text_x = np.clip(0.5 * (dmin + dmax), deg_min, deg_max)
                    text_y = np.clip(hmax, h_min, h_max)
                    ax.text(text_x, text_y, score_text, ha="center", va="bottom", fontsize=9, color=color, bbox=dict(facecolor="black", alpha=0.6))
            elif axis_name in best_box_by_axis:
                dmin, dmax, hmin, hmax = best_box_by_axis[axis_name]
                ax.add_patch(patches.Rectangle((dmin, hmin), dmax - dmin, hmax - hmin, linewidth=2, edgecolor="lime", facecolor="none", alpha=0.9))
                score_text = f"conf={score:.3f}" if use_confidence else f"score={score:.3f}"
                text_x = np.clip(0.5 * (dmin + dmax), deg_min, deg_max)
                text_y = np.clip(hmax, h_min, h_max)
                ax.text(text_x, text_y, score_text, ha="center", va="bottom", fontsize=9, color="lime", bbox=dict(facecolor="black", alpha=0.6))
            if draw_candidates and not top3_by_axis and not draw_only_highlighted:
                confs = conf_by_axis.get(axis_name) if conf_by_axis else None
                for j, b in enumerate(boxes_by_axis.get(axis_name, [])):
                    if use_confidence and confs is not None and float(confs[j]) < confidence_threshold:
                        continue
                    dmin, dmax, hmin, hmax = b
                    ax.add_patch(patches.Rectangle((dmin, hmin), dmax - dmin, hmax - hmin, linewidth=1, edgecolor="yellow", facecolor="none", alpha=0.6))
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"plot 저장 실패 {output_file}: {e}") from e
    plt.close(fig)


def _compute_sample_boxes_top3(
    i: int,
    csv_path: str,
    meta: dict,
    label: int,
    preds_by_axis: Dict,
    preds_conf_by_axis: Dict,
    min_box_ratio: float,
    min_box_degree_span: float,
    min_box_height_span: float,
    has_confidence: bool,
) -> Tuple[Dict, Dict, Dict, Dict, Optional[Dict], Optional[Dict]]:
    """샘플 i에 대해 boxes_by_axis, best_box_by_axis, score_by_axis, top3_by_axis, data_extent, conf_map 반환."""
    fmm = meta["feature_min_max"]
    df_csv = load_crop_csv(csv_path)
    data_extent = None
    if df_csv is not None and not df_csv.empty and "height" in df_csv.columns and "degree" in df_csv.columns:
        data_extent = {
            "degree_min": float(df_csv["degree"].min()),
            "degree_max": float(df_csv["degree"].max()),
            "height_min": float(df_csv["height"].min()),
            "height_max": float(df_csv["height"].max()),
        }
    boxes_by_axis = {}
    best_box_by_axis = {}
    score_by_axis = {}
    top3_by_axis = {}
    for axis in AXIS_NAMES:
        if axis not in preds_by_axis:
            continue
        pred_boxes = preds_by_axis[axis][i]
        conf_arr = preds_conf_by_axis.get(axis)
        boxes_phys = [decode_bbox_norm_to_physical(b, fmm) for b in pred_boxes]
        boxes_by_axis[axis] = boxes_phys
        if conf_arr is not None:
            confs = conf_arr[i] if hasattr(conf_arr, "ndim") and conf_arr.ndim > 1 else conf_arr
            ranked_all = [(boxes_phys[j], float(confs[j])) for j in np.argsort(confs)[::-1]]
        else:
            best_idx, score = select_consistent_box(pred_boxes)
            if pred_boxes.shape[0] > 1:
                iou_mat = iou_matrix(pred_boxes)
                iou_sum = (iou_mat.sum(axis=1) - np.diag(iou_mat)) / (pred_boxes.shape[0] - 1)
                ranked_all = [(boxes_phys[j], float(iou_sum[j])) for j in np.argsort(iou_sum)[::-1]]
            else:
                ranked_all = [(boxes_phys[0], score)] if boxes_phys else []
        ranked_top5 = ranked_all[:5]
        filtered = [
            (box, s) for box, s in ranked_top5
            if _box_passes_size_filter(box, data_extent, min_box_ratio, min_box_degree_span, min_box_height_span)
        ]
        top3_by_axis[axis] = filtered[:3]
        if filtered:
            best_box_by_axis[axis] = filtered[0][0]
            score_by_axis[axis] = filtered[0][1]
    conf_map = {ax: preds_conf_by_axis[ax][i] for ax in AXIS_NAMES if ax in preds_conf_by_axis and preds_conf_by_axis[ax] is not None} if has_confidence else None
    return boxes_by_axis, best_box_by_axis, score_by_axis, top3_by_axis, data_extent, conf_map


def load_models(models_dir: Path) -> Dict[str, keras.Model]:
    models = {}
    for axis in AXIS_NAMES:
        ckpt = models_dir / "checkpoints" / f"best_{axis}.keras"
        if ckpt.exists():
            models[axis] = keras.models.load_model(str(ckpt), compile=False)
    return models


def load_conf_models(models_dir: Path) -> Dict[str, keras.Model]:
    """
    별도로 학습한 confidence head(conf_x/y/z.keras)를 로드.
    없으면 빈 dict 반환.
    """
    conf_models = {}
    for axis in AXIS_NAMES:
        ckpt = models_dir / "checkpoints" / f"conf_{axis}.keras"
        if ckpt.exists():
            conf_models[axis] = keras.models.load_model(str(ckpt), compile=False)
    return conf_models


def _compute_eval_metrics(rows: List[dict], has_confidence: bool) -> dict:
    """
    축별·전체 상관계수(Pearson r), F1(IoU≥0.5 vs conf), break/normal conf 분리 지표 계산.
    - pearson_r: per-box (conf, IoU) 쌍의 상관계수
    - best_f1: conf threshold 스윕 시 IoU≥0.5 예측 F1 최대값 및 해당 threshold
    - mean_conf_break / mean_conf_normal: 샘플별 max conf 평균
    - separation: (mean_break - mean_normal) / (std_break + std_normal) — break vs normal 분리도
    - Spearman, AUC는 선택( scipy/sklearn 있으면 계산 )
    """
    break_rows = [r for r in rows if r.get("label") == 1]
    normal_rows = [r for r in rows if r.get("label") == 0]
    out = {"by_axis": {}, "overall": {}}

    try:
        from scipy.stats import spearmanr
        has_spearman = True
    except ImportError:
        has_spearman = False
    try:
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        has_auc = True
        has_f1 = True
    except ImportError:
        has_auc = False
        has_f1 = False

    def _one_axis(axis: Optional[str]) -> dict:
        if axis is None:
            conf_key, iou_key, pairs_key = "max_conf", "max_iou", "box_conf_iou_pairs"
        else:
            conf_key, iou_key, pairs_key = f"{axis}_score", f"{axis}_max_iou", f"{axis}_box_conf_iou_pairs"
        conf_break = [r[conf_key] for r in break_rows if r.get(conf_key) is not None]
        conf_normal = [r[conf_key] for r in normal_rows if r.get(conf_key) is not None]
        pairs = [p for r in break_rows for p in r.get(pairs_key, [])]

        mean_cb = float(np.mean(conf_break)) if conf_break else None
        std_cb = float(np.std(conf_break)) if len(conf_break) > 1 else 0.0
        mean_cn = float(np.mean(conf_normal)) if conf_normal else None
        std_cn = float(np.std(conf_normal)) if len(conf_normal) > 1 else 0.0
        separation = None
        if mean_cb is not None and mean_cn is not None and (std_cb + std_cn) > 1e-8:
            separation = (mean_cb - mean_cn) / (std_cb + std_cn + 1e-8)

        pearson_r, spearman_r, auc_high_iou = None, None, None
        best_f1, best_f1_threshold, best_f1_precision, best_f1_recall = None, None, None, None
        n_pairs = len(pairs)
        if pairs and len(pairs) > 1 and has_confidence:
            confs = np.array([p[0] for p in pairs])
            ious = np.array([p[1] for p in pairs])
            pearson_r = float(np.corrcoef(confs, ious)[0, 1])
            if has_spearman:
                sp = spearmanr(confs, ious)
                spearman_r = float(sp.correlation) if sp.correlation is not None else None
            binary = (ious >= 0.5).astype(np.int32)
            if has_auc and np.unique(binary).size == 2:
                auc_high_iou = float(roc_auc_score(binary, confs))
            elif has_auc:
                auc_high_iou = None
            if has_f1 and np.unique(binary).size == 2:
                ths = np.linspace(0.01, 0.99, 99)
                best_f1, best_th = 0.0, 0.5
                best_pr, best_rc = 0.0, 0.0
                for th in ths:
                    y_pred = (confs >= th).astype(np.int32)
                    f1 = float(f1_score(binary, y_pred, zero_division=0))
                    if f1 >= best_f1:
                        best_f1, best_th = f1, th
                        best_pr = float(precision_score(binary, y_pred, zero_division=0))
                        best_rc = float(recall_score(binary, y_pred, zero_division=0))
                best_f1_threshold = float(best_th)
                best_f1_precision = best_pr
                best_f1_recall = best_rc

        entry = {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "auc_high_iou": auc_high_iou,
            "best_f1": best_f1,
            "best_f1_threshold": best_f1_threshold,
            "best_f1_precision": best_f1_precision,
            "best_f1_recall": best_f1_recall,
            "n_pairs": n_pairs,
            "mean_conf_break": mean_cb,
            "mean_conf_normal": mean_cn,
            "std_conf_break": std_cb if conf_break else None,
            "std_conf_normal": std_cn if conf_normal else None,
            "separation": separation,
            "n_break": len(conf_break),
            "n_normal": len(conf_normal),
        }
        return entry

    for axis in AXIS_NAMES:
        out["by_axis"][axis] = _one_axis(axis)
    out["overall"] = _one_axis(None)
    return out


def _plot_eval_distributions(rows: List[dict], out_dir: Path, has_confidence: bool, axis: Optional[str] = None) -> None:
    """IoU / conf 분포 및 IoU–conf 연관성 그래프를 out_dir에 저장.

    Subplots:
      1) IoU dist (break, w/ GT)
      2) conf dist (break)
      3) conf dist (normal)
      4) IoU vs conf scatter (break, per-box)

    axis=None이면 전체, 'x'/'y'/'z'면 해당 축만."""
    break_rows = [r for r in rows if r.get("label") == 1]
    normal_rows = [r for r in rows if r.get("label") == 0]
    if axis is None:
        iou_key, conf_key = "max_iou", "max_conf"
        axis_label = "All"
    else:
        iou_key = f"{axis}_max_iou"
        conf_key = f"{axis}_score"
        axis_label = f"{axis.upper()} axis"
    iou_vals = [r[iou_key] for r in break_rows if r.get(iou_key) is not None]
    conf_break = [r[conf_key] for r in break_rows if r.get(conf_key) is not None]
    conf_normal = [r[conf_key] for r in normal_rows if r.get(conf_key) is not None]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    suptitle = f"Distributions (axis: {axis_label})"
    fig.suptitle(suptitle, fontsize=11, y=1.02)

    # IoU dist. (break samples, w/ GT)
    ax = axes[0]
    if iou_vals:
        ax.hist(iou_vals, bins=min(30, max(10, len(iou_vals) // 5)), color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("max IoU" if axis is None else f"{axis.upper()} max IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU dist. (break, w/ GT)")
    ax.grid(True, alpha=0.3)

    # conf dist. (break)
    ax = axes[1]
    if conf_break:
        ax.hist(
            conf_break,
            bins=min(30, max(10, len(conf_break) // 5)),
            color="coral",
            alpha=0.8,
            edgecolor="white",
        )
    mean_cb = np.mean(conf_break) if conf_break else None
    ax.set_xlabel("max conf (Break)" if axis is None else f"{axis.upper()} conf (Break)")
    ax.set_ylabel("Count")
    title_cb = "conf dist. (break)"
    if mean_cb is not None:
        title_cb += f"\nmean={mean_cb:.3f}"
    ax.set_title(title_cb)
    ax.grid(True, alpha=0.3)

    # conf dist. (normal)
    ax = axes[2]
    if conf_normal:
        ax.hist(
            conf_normal,
            bins=min(30, max(10, len(conf_normal) // 5)),
            color="seagreen",
            alpha=0.8,
            edgecolor="white",
        )
    mean_cn = np.mean(conf_normal) if conf_normal else None
    ax.set_xlabel("max conf (Normal)" if axis is None else f"{axis.upper()} conf (Normal)")
    ax.set_ylabel("Count")
    title_cn = "conf dist. (normal)"
    if mean_cn is not None:
        title_cn += f"\nmean={mean_cn:.3f}"
    ax.set_title(title_cn)
    ax.grid(True, alpha=0.3)

    # IoU vs conf scatter: 박스당 (conf, iou) 한 점 (break, GT 있는 박스만)
    ax = axes[3]
    if axis is None:
        pairs = [p for r in break_rows for p in r.get("box_conf_iou_pairs", [])]
    else:
        pairs = [p for r in break_rows for p in r.get(f"{axis}_box_conf_iou_pairs", [])]
    if pairs and has_confidence:
        conf_for_iou = np.array([p[0] for p in pairs])
        iou_for_scatter = np.array([p[1] for p in pairs])
        ax.scatter(conf_for_iou, iou_for_scatter, alpha=0.5, s=12, c="steelblue", edgecolors="none")
        if len(iou_for_scatter) > 1:
            r_val = float(np.corrcoef(conf_for_iou, iou_for_scatter)[0, 1])
            txt = f"Pearson r={r_val:.3f} (n={len(pairs)})"
            try:
                from scipy.stats import spearmanr
                sp = spearmanr(conf_for_iou, iou_for_scatter)
                if sp.correlation is not None:
                    txt += f"\nSpearman ρ={sp.correlation:.3f}"
            except ImportError:
                pass
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va="top")
        ax.set_xlabel("conf (per box)" if axis is None else f"{axis.upper()} conf (per box)")
        ax.set_ylabel("IoU (per box)" if axis is None else f"{axis.upper()} IoU (per box)")
        ax.set_title("IoU vs conf (per box, break)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No IoU/conf data\n(break + 5*P model)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("IoU vs conf (per box)")

    plt.tight_layout()
    fname = "eval_distributions.png" if axis is None else f"eval_distributions_{axis}.png"
    out_path = out_dir / fname
    try:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        plt.close(fig)
        print(f"경고: 분포 그래프 저장 실패 {out_path}: {e}")


def _plot_f1_threshold_sweep(rows: List[dict], out_dir: Path, has_confidence: bool) -> None:
    """conf threshold 스윕 시 Precision / Recall / F1 곡선을 2x2(축별 + overall)로 저장."""
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
    except ImportError:
        return
    break_rows = [r for r in rows if r.get("label") == 1]
    configs = [
        (None, "max_conf", "box_conf_iou_pairs", "Overall"),
        ("x", "x_score", "x_box_conf_iou_pairs", "X axis"),
        ("y", "y_score", "y_box_conf_iou_pairs", "Y axis"),
        ("z", "z_score", "z_box_conf_iou_pairs", "Z axis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    ths = np.linspace(0.01, 0.99, 99)

    for ax_plot, (axis, conf_key, pairs_key, title) in zip(axes_flat, configs):
        pairs = [p for r in break_rows for p in r.get(pairs_key, [])]
        if not pairs or not has_confidence:
            ax_plot.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_plot.transAxes)
            ax_plot.set_title(title)
            ax_plot.set_xlabel("conf threshold")
            continue
        confs = np.array([p[0] for p in pairs])
        ious = np.array([p[1] for p in pairs])
        binary = (ious >= 0.5).astype(np.int32)
        if np.unique(binary).size != 2:
            ax_plot.text(0.5, 0.5, "IoU≥0.5 / <0.5 둘 다 필요", ha="center", va="center", transform=ax_plot.transAxes)
            ax_plot.set_title(title)
            ax_plot.set_xlabel("conf threshold")
            continue
        precs, recs, f1s = [], [], []
        for th in ths:
            y_pred = (confs >= th).astype(np.int32)
            precs.append(float(precision_score(binary, y_pred, zero_division=0)))
            recs.append(float(recall_score(binary, y_pred, zero_division=0)))
            f1s.append(float(f1_score(binary, y_pred, zero_division=0)))
        ax_plot.plot(ths, precs, label="Precision", color="C0")
        ax_plot.plot(ths, recs, label="Recall", color="C1")
        ax_plot.plot(ths, f1s, label="F1", color="C2")
        ax_plot.set_xlim(0, 1)
        ax_plot.set_ylim(0, 1.02)
        ax_plot.set_xlabel("conf threshold")
        ax_plot.set_ylabel("Score")
        ax_plot.set_title(f"{title} (n={len(pairs)})")
        ax_plot.legend()
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle("Precision / Recall / F1 vs conf threshold (IoU≥0.5 = positive)", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "f1_threshold_sweep.png"
    try:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {out_path}")
    except Exception as e:
        plt.close(fig)
        print(f"경고: F1 threshold sweep 저장 실패 {out_path}: {e}")


def _plot_precision_recall_curve(rows: List[dict], out_dir: Path, has_confidence: bool) -> None:
    """Precision–Recall 곡선을 축별 + overall 2x2로 저장 (IoU≥0.5 = positive, conf = score)."""
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        return
    break_rows = [r for r in rows if r.get("label") == 1]
    configs = [
        (None, "box_conf_iou_pairs", "Overall"),
        ("x", "x_box_conf_iou_pairs", "X axis"),
        ("y", "y_box_conf_iou_pairs", "Y axis"),
        ("z", "z_box_conf_iou_pairs", "Z axis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax_plot, (axis, pairs_key, title) in zip(axes_flat, configs):
        pairs = [p for r in break_rows for p in r.get(pairs_key, [])]
        if not pairs or not has_confidence:
            ax_plot.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_plot.transAxes)
            ax_plot.set_title(title)
            ax_plot.set_xlabel("Recall")
            ax_plot.set_ylabel("Precision")
            continue
        confs = np.array([p[0] for p in pairs])
        ious = np.array([p[1] for p in pairs])
        binary = (ious >= 0.5).astype(np.int32)
        if np.unique(binary).size != 2:
            ax_plot.text(0.5, 0.5, "IoU≥0.5 / <0.5 둘 다 필요", ha="center", va="center", transform=ax_plot.transAxes)
            ax_plot.set_title(title)
            ax_plot.set_xlabel("Recall")
            ax_plot.set_ylabel("Precision")
            continue
        precision, recall, _ = precision_recall_curve(binary, confs)
        ap = float(average_precision_score(binary, confs))
        ax_plot.plot(recall, precision, color="C0", lw=2)
        ax_plot.set_xlim(0, 1.02)
        ax_plot.set_ylim(0, 1.02)
        ax_plot.set_xlabel("Recall")
        ax_plot.set_ylabel("Precision")
        ax_plot.set_title(f"{title} (n={len(pairs)}, AP={ap:.3f})")
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle("Precision–Recall curve (IoU≥0.5 = positive, conf = score)", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "precision_recall_curve.png"
    try:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {out_path}")
    except Exception as e:
        plt.close(fig)
        print(f"경고: Precision–Recall 곡선 저장 실패 {out_path}: {e}")


def _plot_training_curves_2nd(second_stage_run_dir: Path, out_dir: Path) -> None:
    """
    2차 Hard 모델(conf head) 학습 곡선을
    12. hard_models_2nd/<run>/histories.json 에서 읽어 out_dir/training_curves.png 로 저장.
    """
    hist_path = second_stage_run_dir / "histories.json"
    if not hist_path.exists():
        return
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            histories = json.load(f)
    except Exception:
        return
    if not isinstance(histories, dict) or not histories:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, axis_name in enumerate(AXIS_NAMES):
        h = histories.get(axis_name, {})
        if not isinstance(h, dict) or not h:
            continue
        epochs = range(1, len(h.get("loss", [])) + 1)

        # 1행: loss / val_loss
        ax_loss = axes[0, idx]
        loss_vals = h.get("loss", [])
        val_loss_vals = h.get("val_loss", [])
        if loss_vals:
            ax_loss.plot(epochs[: len(loss_vals)], loss_vals, label="loss")
        if val_loss_vals:
            ax_loss.plot(epochs[: len(val_loss_vals)], val_loss_vals, label="val_loss")
        ax_loss.set_title(f"{axis_name.upper()} loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.3)
        if loss_vals or val_loss_vals:
            ax_loss.legend()

        # 2행: accuracy / val_accuracy
        ax_acc = axes[1, idx]
        acc_vals = h.get("accuracy", [])
        val_acc_vals = h.get("val_accuracy", [])
        if acc_vals:
            ax_acc.plot(epochs[: len(acc_vals)], acc_vals, label="accuracy")
        if val_acc_vals:
            ax_acc.plot(epochs[: len(val_acc_vals)], val_acc_vals, label="val_accuracy")
        ax_acc.set_title(f"{axis_name.upper()} accuracy")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.grid(True, alpha=0.3)
        if acc_vals or val_acc_vals:
            ax_acc.legend()

    fig.suptitle("2nd-stage conf head training curves (12. hard_models_2nd)", y=1.02)
    plt.tight_layout()
    curves_path = out_dir / "training_curves.png"
    try:
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("저장:", curves_path)
    except Exception as e:
        plt.close(fig)
        print(f"경고: 학습 곡선 저장 실패 {curves_path}: {e}")


def build_dataset(data_dir: str, only_break: bool, only_normal: bool, min_points: int, max_points: int, max_files: Optional[int]):
    imgs, metas, csv_paths, labels = [], [], [], []
    files = []
    if not only_normal:
        files += collect_all_crop_files(data_dir, True)
    if not only_break:
        files += collect_all_crop_files(data_dir, False)
    if max_files:
        files = files[:max_files]
    for csv_path, _, _, label in tqdm(files, desc="데이터 전처리"):
        r = prepare_sequence_from_csv(csv_path=csv_path, sort_by="height", feature_min_max=None)
        if r is None:
            continue
        img, meta = r
        if not (min_points <= meta.get("original_length", 0) <= max_points):
            continue
        img = resize_img_height(img, target_h=IMG_HEIGHT)
        imgs.append(img)
        metas.append(meta)
        csv_paths.append(csv_path)
        labels.append(int(label))
    if not imgs:
        return np.zeros((0, IMG_HEIGHT, 19, 3), dtype=np.float32), [], [], []
    return np.array(imgs, dtype=np.float32), metas, csv_paths, labels


def _process_one_sample_sync(
    i: int,
    csv_path: str,
    meta: dict,
    label: int,
    preds_by_axis: Dict,
    preds_conf_by_axis: Dict,
    break_dir: Path,
    normal_dir: Path,
    min_box_ratio: float,
    min_box_degree_span: float,
    min_box_height_span: float,
    has_confidence: bool,
    gt_from_npy: Optional[Dict[str, Dict[str, List[Tuple[float, float, float, float]]]]] = None,
) -> dict:
    """한 샘플에 대해 박스 계산, JSON 저장, row 반환. gt_from_npy가 있으면 ROI JSON 없을 때 9. hard_train_data NPY GT 사용."""
    # CSV 경로 전체가 sample_id로 들어가는 것을 방지하고,
    # 파일명 기반의 안정적인 ID를 사용한다.
    raw_id = _get_sample_id_from_csv(csv_path) or Path(csv_path).stem.replace("_OUT_processed", "")
    # 혹시라도 경로가 들어온 경우(절대/상대 경로 포함)를 대비해 마지막 파일명만 남기고 정규화
    sid = str(raw_id).replace("\\", "/")
    sid = sid.split("/")[-1]
    # 확장자가 남아 있는 경우 제거 (예: xxx_OUT_processed.csv)
    if sid.lower().endswith(".csv"):
        sid = sid[:-4]
    sample_id = sid
    fmm = meta["feature_min_max"]
    df_csv = load_crop_csv(csv_path)
    data_extent = None
    if df_csv is not None and not df_csv.empty and "height" in df_csv.columns and "degree" in df_csv.columns:
        data_extent = {
            "degree_min": float(df_csv["degree"].min()),
            "degree_max": float(df_csv["degree"].max()),
            "height_min": float(df_csv["height"].min()),
            "height_max": float(df_csv["height"].max()),
        }
    boxes_by_axis = {}
    best_box_by_axis = {}
    score_by_axis = {}
    top3_by_axis = {}
    has_drawable_box = False
    for axis in AXIS_NAMES:
        if axis not in preds_by_axis:
            continue
        pred_boxes = preds_by_axis[axis][i]
        conf_arr = preds_conf_by_axis.get(axis)
        boxes_phys = [decode_bbox_norm_to_physical(b, fmm) for b in pred_boxes]
        boxes_by_axis[axis] = boxes_phys
        if conf_arr is not None:
            confs = conf_arr[i]
            ranked_all = [(boxes_phys[j], float(confs[j])) for j in np.argsort(confs)[::-1]]
        else:
            best_idx, score = select_consistent_box(pred_boxes)
            if pred_boxes.shape[0] > 1:
                iou_mat = iou_matrix(pred_boxes)
                iou_sum = (iou_mat.sum(axis=1) - np.diag(iou_mat)) / (pred_boxes.shape[0] - 1)
                ranked_all = [(boxes_phys[j], float(iou_sum[j])) for j in np.argsort(iou_sum)[::-1]]
            else:
                ranked_all = [(boxes_phys[0], score)] if boxes_phys else []
        ranked_top5 = ranked_all[:5]
        filtered = [
            (box, s) for box, s in ranked_top5
            if _box_passes_size_filter(box, data_extent, min_box_ratio, min_box_degree_span, min_box_height_span)
        ]
        top3_by_axis[axis] = filtered[:3]
        if filtered:
            has_drawable_box = True
            best_box_by_axis[axis] = filtered[0][0]
            score_by_axis[axis] = filtered[0][1]
    label_dir = break_dir if label == 1 else normal_dir
    gt_boxes_by_axis = {}
    if label == 1:
        roi_path = match_roi_json_from_csv(csv_path)
        if roi_path and os.path.exists(roi_path):
            roi_info = load_roi_info_json(roi_path)
            if roi_info:
                expanded = expand_rois_from_roi_info(roi_info)
                for axis_idx, axis_name in enumerate(AXIS_NAMES):
                    gt_list = []
                    for rid, b in expanded:
                        if rid != axis_idx:
                            continue
                        hc, hw, dc, dw = b[0], b[1], b[2], b[3]
                        gt_list.append((dc - dw / 2.0, dc + dw / 2.0, hc - hw / 2.0, hc + hw / 2.0))
                    if gt_list:
                        gt_boxes_by_axis[axis_name] = gt_list
        if not gt_boxes_by_axis and gt_from_npy:
            for key in (str(Path(csv_path).resolve()), csv_path):
                if key in gt_from_npy:
                    gt_boxes_by_axis = gt_from_npy[key].copy()
                    break
    boxes_json = {"sample_id": sample_id, "label": int(label), "data_extent": data_extent, "axes": {}}
    for axis in AXIS_NAMES:
        if axis not in top3_by_axis:
            continue
        ranked = top3_by_axis[axis]
        gt_boxes = gt_boxes_by_axis.get(axis)
        boxes_json["axes"][axis] = []
        for rank, (box, score) in enumerate(ranked):
            dmin, dmax, hmin, hmax = box
            entry = {"rank": rank + 1, "degree_min": round(dmin, 6), "degree_max": round(dmax, 6), "height_min": round(hmin, 6), "height_max": round(hmax, 6), "score": round(float(score), 6)}
            if gt_boxes:
                entry["iou"] = round(max(box_iou_physical(box, gt) for gt in gt_boxes), 6)
            if data_extent is not None:
                deg_span = max(data_extent["degree_max"] - data_extent["degree_min"], 1e-9)
                h_span = max(data_extent["height_max"] - data_extent["height_min"], 1e-9)
                box_deg_span = max(dmax - dmin, 0)
                box_h_span = max(hmax - hmin, 0)
                entry["inside_data_extent"] = (data_extent["degree_min"] <= dmin <= data_extent["degree_max"] and data_extent["degree_min"] <= dmax <= data_extent["degree_max"] and data_extent["height_min"] <= hmin <= data_extent["height_max"] and data_extent["height_min"] <= hmax <= data_extent["height_max"])
                entry["below_min_size"] = (min_box_ratio > 0 and ((box_deg_span / deg_span) < min_box_ratio or (box_h_span / h_span) < min_box_ratio)) or (min_box_degree_span > 0 and box_deg_span < min_box_degree_span) or (min_box_height_span > 0 and box_h_span < min_box_height_span)
            boxes_json["axes"][axis].append(entry)
    with open(label_dir / f"{sample_id}_pred_boxes.json", "w", encoding="utf-8") as f:
        json.dump(boxes_json, f, ensure_ascii=False, indent=2)
    max_conf = max(score_by_axis.values()) if score_by_axis else None
    iou_vals = [entry["iou"] for axis in boxes_json.get("axes", {}) for entry in boxes_json["axes"][axis] if "iou" in entry]
    max_iou = max(iou_vals) if iou_vals else None
    row = {"sample_id": sample_id, "csv_path": csv_path, "label": int(label), "max_conf": max_conf, "max_iou": max_iou, "has_drawable_box": has_drawable_box}
    # 박스당 (conf, iou) 쌍: break + GT 있을 때만 (eval_distributions 세 번째 그래프용)
    box_conf_iou_pairs = []
    for ax in AXIS_NAMES:
        if ax not in boxes_json.get("axes", {}):
            continue
        ax_pairs = []
        for e in boxes_json["axes"][ax]:
            if "score" in e and "iou" in e:
                box_conf_iou_pairs.append((float(e["score"]), float(e["iou"])))
                ax_pairs.append((float(e["score"]), float(e["iou"])))
        row[f"{ax}_box_conf_iou_pairs"] = ax_pairs
    row["box_conf_iou_pairs"] = box_conf_iou_pairs
    for ax in AXIS_NAMES:
        if ax in boxes_json.get("axes", {}):
            iou_ax = [e["iou"] for e in boxes_json["axes"][ax] if "iou" in e]
            row[f"{ax}_max_iou"] = max(iou_ax) if iou_ax else None
            row[f"{ax}_has_box"] = len(boxes_json["axes"][ax]) > 0
        else:
            row[f"{ax}_max_iou"] = None
            row[f"{ax}_has_box"] = False
            if f"{ax}_box_conf_iou_pairs" not in row:
                row[f"{ax}_box_conf_iou_pairs"] = []
    for axis in AXIS_NAMES:
        if axis in best_box_by_axis:
            dmin, dmax, hmin, hmax = best_box_by_axis[axis]
            row.update({f"{axis}_degree_min": dmin, f"{axis}_degree_max": dmax, f"{axis}_height_min": hmin, f"{axis}_height_max": hmax, f"{axis}_score": score_by_axis.get(axis, 0)})
    row["_idx"] = i  # 요약 이미지 생성 시 경로 매칭 없이 인덱스로 바로 사용 (Windows/WSL 경로 차이 회피)
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hard 2차(conf) 모델 평가")
    parser.add_argument("--data-dir", default="4. merge_data")
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--only-break", action="store_true")
    parser.add_argument("--only-normal", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU 추론 배치 크기")
    parser.add_argument("--repreprocess", action="store_true", help="전처리 캐시 무시하고 매번 전처리")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="5*P 모델: 이 값 이상인 박스만 표시")
    parser.add_argument("--min-box-ratio", type=float, default=0.02, help="데이터 범위 대비 최소 박스 크기 비율. 이 비율 미만인 박스는 그리지 않음 (기본 0.02=2%%)")
    parser.add_argument("--min-box-degree-span", type=float, default=5.0, help="최소 박스 degree 폭(물리 단위). 0이면 미적용 (기본 5)")
    parser.add_argument("--min-box-height-span", type=float, default=0.1, help="최소 박스 height 폭(물리 단위). 0이면 미적용 (기본 0.1)")
    parser.add_argument("--test-only", action="store_true", default=True, help="9. hard_train_data의 test 샘플만 평가 (기본값: True, 전체 데이터 평가하려면 --no-test-only 사용)")
    parser.add_argument("--no-test-only", dest="test_only", action="store_false", help="전체 데이터 평가 (test 필터링 비활성화)")
    parser.add_argument("--run-dir", type=str, default=None, help="평가할 2차 run 디렉터리")
    parser.add_argument("--target-overall-best-f1", type=float, default=0.70, help="재학습 기준: overall best_f1")
    parser.add_argument("--target-overall-auc", type=float, default=0.70, help="재학습 기준: overall AUC")
    parser.add_argument("--target-overall-separation", type=float, default=0.20, help="재학습 기준: overall separation")
    parser.add_argument(
        "--skip-sample-images",
        action="store_true",
        help="샘플별 결과 이미지(test_result_img, info 하위 PNG) 생성을 건너뛰어 평가 시간을 단축",
    )
    parser.add_argument(
        "--target-pass-mode",
        type=str,
        choices=["all_metrics", "f1_only"],
        default="all_metrics",
        help="재학습 판정 모드",
    )
    args = parser.parse_args()

    # 2차 Hard 모델(Conf head) 최근 run 및 해당 1차 Hard 모델 run 정보 로드
    if args.run_dir is not None:
        second_stage_run_dir = Path(args.run_dir).resolve()
        if not second_stage_run_dir.exists():
            raise FileNotFoundError(f"--run-dir path does not exist: {second_stage_run_dir}")
    else:
        second_stage_run_dir = _get_latest_second_stage_run(SECOND_STAGE_BASE)
    if second_stage_run_dir is None:
        raise FileNotFoundError(f"2nd-stage hard model run not found under: {SECOND_STAGE_BASE}")

    print(f"Evaluating 2nd-stage hard model run: {second_stage_run_dir.name}")
    training_config_path = second_stage_run_dir / "training_config.json"
    second_stage_config = {}
    first_stage_run_dir = None
    if training_config_path.exists():
        with open(training_config_path, "r", encoding="utf-8") as f:
            second_stage_config = json.load(f)
        first_stage_ckpt_dir_str = second_stage_config.get("first_stage_ckpt_dir")
        if first_stage_ckpt_dir_str:
            first_stage_run_dir = Path(first_stage_ckpt_dir_str).parent
    if first_stage_run_dir is None or not first_stage_run_dir.is_dir():
        # fallback: 가장 최근 1차 hard run 사용
        print("first_stage_run_dir를 training_config에서 찾지 못해 최신 1st-stage run으로 대체합니다.")
        first_stage_runs = [d for d in FIRST_STAGE_BASE.iterdir() if d.is_dir()]
        if not first_stage_runs:
            raise FileNotFoundError(f"1st-stage hard model run not found under: {FIRST_STAGE_BASE}")
        first_stage_run_dir = max(first_stage_runs, key=lambda d: d.name)

    print(f"Using 1st-stage hard model run: {first_stage_run_dir.name} ({first_stage_run_dir})")

    models_dir = first_stage_run_dir
    data_dir = Path(current_dir) / args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 경로 없음: {data_dir}")

    models = load_models(models_dir)
    if not models:
        raise FileNotFoundError("모델 없음")

    # 2차 평가용 전처리 캐시는 9. hard_train_data/<run_subdir>/eval_cache 아래에 저장된 것을 우선 사용
    hard_data_base = Path(current_dir) / "9. hard_train_data"
    eval_cache_dirs = []
    if hard_data_base.exists():
        for d in hard_data_base.iterdir():
            if d.is_dir():
                ec = d / "eval_cache"
                if ec.exists() and ec.is_dir():
                    eval_cache_dirs.append(ec)
    if eval_cache_dirs:
        cache_dir = max(eval_cache_dirs, key=lambda p: p.parent.name)
    else:
        # fallback: 예전 경로(존재할 수도 있으므로 유지)
        cache_dir = Path(current_dir) / "11. evaluate_resnet_model" / "preprocessed_cache"
    cache_key = f"{args.data_dir}_{args.min_points}_{args.max_points}_{args.only_break}_{args.only_normal}_{args.max_files}".replace(" ", "_").replace(".", "_")
    cache_npz = cache_dir / f"{cache_key}.npz"
    cache_pkl = cache_dir / f"{cache_key}.pkl"

    if not args.repreprocess and cache_npz.exists() and cache_pkl.exists():
        print("전처리 캐시 로드:", cache_npz)
        npz = np.load(cache_npz)
        X = npz["X"]
        with open(cache_pkl, "rb") as f:
            metas, csv_paths, labels = pickle.load(f)
    else:
        X, metas, csv_paths, labels = build_dataset(
            str(args.data_dir), args.only_break, args.only_normal,
            args.min_points, args.max_points, args.max_files,
        )
        if len(X) > 0:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # cache_key에 경로 구분자가 있을 수 있으므로 cache_npz의 부모 디렉터리도 생성
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_npz, X=X)
            cache_pkl.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_pkl, "wb") as f:
                pickle.dump((metas, csv_paths, labels), f)
            print("전처리 결과 저장:", cache_npz)

    if len(X) == 0:
        print("유효한 데이터 없음")
        return

    # 9. hard_train_data 경로
    hard_data_base = Path(current_dir) / "9. hard_train_data"
    
    # --test-only: 9. hard_train_data의 test_indices로 샘플 필터링
    if args.test_only:
        if not hard_data_base.exists():
            print(f"경고: --test-only 지정했으나 9. hard_train_data 없음. 전체 데이터 사용")
        else:
            # 최신 run의 metadata 로드
            hard_runs = sorted([d for d in hard_data_base.iterdir() if d.is_dir()], key=lambda d: d.name)
            if not hard_runs:
                print(f"경고: --test-only 지정했으나 9. hard_train_data 아래 run 없음. 전체 데이터 사용")
            else:
                latest_hard_run = hard_runs[-1]
                metadata_path = latest_hard_run / "break_imgs_metadata.json"
                if not metadata_path.exists():
                    print(f"경고: {metadata_path} 없음. 전체 데이터 사용")
                else:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        hard_metadata = json.load(f)
                    test_indices = hard_metadata.get("test_indices", [])
                    if not test_indices:
                        print(f"경고: metadata에 test_indices 없음. 전체 데이터 사용")
                    else:
                        # test_indices에 해당하는 csv_path 추출
                        test_csv_paths = set()
                        samples = hard_metadata.get("samples", [])
                        for idx in test_indices:
                            if idx < len(samples):
                                csv_path = samples[idx].get("csv_path")
                                if csv_path:
                                    # 경로 정규화 (Windows/Linux 호환)
                                    test_csv_paths.add(str(Path(csv_path).resolve()))
                        
                        # 현재 csv_paths와 매칭
                        keep_indices = []
                        for i, cp in enumerate(csv_paths):
                            cp_resolved = str(Path(cp).resolve())
                            if cp_resolved in test_csv_paths:
                                keep_indices.append(i)
                        
                        if keep_indices:
                            X = X[keep_indices]
                            metas = [metas[i] for i in keep_indices]
                            csv_paths = [csv_paths[i] for i in keep_indices]
                            labels = [labels[i] for i in keep_indices]
                            print(f"--test-only: {len(keep_indices)}개 테스트 샘플만 평가 (9. hard_train_data/{latest_hard_run.name})")
                        else:
                            print(f"경고: test_indices와 매칭되는 샘플 없음. 전체 데이터 사용")
    
    # 9. hard_train_data run의 eval_cache 사용 시, 해당 run의 NPY에서 GT bbox 로드 (ROI JSON 없을 때 IoU 계산용)
    gt_from_npy = None
    if hard_data_base.exists() and cache_dir.parent.parent == hard_data_base and (cache_dir.parent / "break_labels.npy").exists():
        gt_from_npy = load_gt_bbox_from_hard_run(cache_dir.parent)
        if gt_from_npy:
            print("GT bbox 로드 (9. hard_train_data run NPY):", len(gt_from_npy), "break samples")

    # 평가 결과 디렉토리:
    #  - 13. evaluate_hard_model_2nd/<second_stage_run>/test/...  (테스트 데이터 기준 결과)
    #  - 13. evaluate_hard_model_2nd/<second_stage_run>/all/...   (전체 데이터 기준 결과)
    eval_base = Path(current_dir) / "13. evaluate_hard_model_2nd"
    out_dir = eval_base / second_stage_run_dir.name
    scope_name = "test" if args.test_only else "all"
    scope_dir = out_dir / scope_name
    if scope_dir.exists():
        shutil.rmtree(scope_dir)

    images_dir = scope_dir / "info"
    break_dir = images_dir / "break"
    normal_dir = images_dir / "normal"
    summary_dir = scope_dir / "test_result_img"
    summary_categories = []
    # 축별 confidence 구간(0.0~0.1, ..., 0.9~1.0)별로 논리적 카테고리 이름을 만든다.
    # 실제 디렉토리 구조는 conf_<bin_from>_<bin_to>/<axis>/ 형태로 생성한다.
    # 예: test_result_img/conf_0_1/x, test_result_img/conf_0_1/y, test_result_img/conf_0_1/z
    for axis in AXIS_NAMES:
        for bin_idx in range(10):
            summary_categories.append(f"conf_bin_{axis}_{bin_idx}")

    # 기본 디렉토리 + conf 범위별/축별 디렉토리 생성
    dirs_to_make = [out_dir, scope_dir, images_dir, break_dir, normal_dir, summary_dir]
    for bin_idx in range(10):
        bin_dir_name = f"conf_{bin_idx}_{bin_idx + 1}"
        for axis in AXIS_NAMES:
            dirs_to_make.append(summary_dir / bin_dir_name / axis)
    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    preds_by_axis: Dict[str, np.ndarray] = {}
    preds_conf_by_axis: Dict[str, Optional[np.ndarray]] = {}
    has_confidence = False
    P = None
    print(f"GPU 추론 중 (bbox, batch_size={args.batch_size}, 샘플 수={len(X)})...")
    for axis, model in models.items():
        pred = np.asarray(model.predict(X, batch_size=args.batch_size, verbose=0))
        n_out = pred.shape[1]
        if n_out % 5 == 0:
            P = n_out // 5
            pf = pred.reshape(-1, P, 5)
            preds_by_axis[axis] = pf[:, :, :4]
            # conf는 별도 head(conf_x/y/z)가 있으면 그 값을 사용하므로 여기서는 일단 무시
            preds_conf_by_axis[axis] = None
        else:
            P = n_out // 4
            preds_by_axis[axis] = pred.reshape(-1, P, 4)
            preds_conf_by_axis[axis] = None

    # 별도 conf head(conf_x/y/z)가 있으면 그걸 사용해 confidence 예측
    # -> conf head는 2차 Hard run 디렉터리(12. hard_models_2nd/<run>/checkpoints)에 저장되어 있으므로,
    #    first-stage run 이 아니라 second_stage_run_dir 기준으로 로드한다.
    conf_models = load_conf_models(second_stage_run_dir)
    if conf_models:
        print(f"별도 conf head 사용: {list(conf_models.keys())}")
        for axis, c_model in conf_models.items():
            conf_pred = np.asarray(c_model.predict(X, batch_size=args.batch_size, verbose=0))
            # shape: (N, P)
            preds_conf_by_axis[axis] = conf_pred
        has_confidence = True
    else:
        # conf head가 없으면, 5*P 출력(신형 모델)의 마지막 채널을 confidence로 사용 (구동 방식 유지)
        for axis in AXIS_NAMES:
            arr = preds_by_axis.get(axis)
            # 위 루프에서 5*P 모델이면 preds_conf_by_axis[axis]를 None으로 두었으므로,
            # 여기서는 has_confidence=False 그대로 두고, 이후 로직에서 conf 미사용 경로로 동작.
            _ = arr  # 형식상 참조만

    n_samples = len(csv_paths)
    rows = []
    for i in tqdm(range(n_samples), desc="JSON 저장"):
        row = _process_one_sample_sync(
            i, csv_paths[i], metas[i], labels[i],
            preds_by_axis, preds_conf_by_axis,
            break_dir, normal_dir,
            args.min_box_ratio, args.min_box_degree_span, args.min_box_height_span,
            has_confidence,
            gt_from_npy=gt_from_npy,
        )
        rows.append(row)

    # confidence 구간(0.1 단위)별로 축별 요약 이미지를 생성.
    # 박스 그리기 조건을 하나라도 만족하는 샘플만 후보에 포함.
    # has_drawable_box=True 인 샘플만 대상
    rankable_rows = [r for r in rows if r.get("has_drawable_box")]

    # 축별 confidence 0.1 단위 bin 구성
    # bin 0: [0.0, 0.1), bin 1: [0.1, 0.2), ..., bin 8: [0.8, 0.9), bin 9: [0.9, 1.0]
    category_lists = {cat: [] for cat in summary_categories}
    for r in rankable_rows:
        for axis in AXIS_NAMES:
            if not r.get(f"{axis}_has_box"):
                continue
            score = r.get(f"{axis}_score")
            if score is None:
                continue
            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            if s < 0.0 or s > 1.0:
                # 정규화 범위를 벗어난 경우는 스킵
                continue
            bin_idx = int(s * 10.0)
            if bin_idx >= 10:
                bin_idx = 9
            cat = f"conf_bin_{axis}_{bin_idx}"
            category_lists.setdefault(cat, []).append(r)
    summary_lists_json = {k: [r["sample_id"] for r in v] for k, v in category_lists.items()}
    with open(summary_dir / "summary_lists.json", "w", encoding="utf-8") as f:
        json.dump(summary_lists_json, f, ensure_ascii=False, indent=2)

    # ============================================================================
    # conf bin별 통계 및 파단 판정 기준 지표
    # ============================================================================
    conf_bin_stats = {}
    conf_threshold_stats = {}
    for axis in AXIS_NAMES:
        # 해당 축에서 has_box=True인 샘플만
        axis_rows = [r for r in rows if r.get(f"{axis}_has_box")]
        if not axis_rows:
            continue
        
        # bin별 통계 (0.1 단위)
        bin_stats = []
        for bin_idx in range(10):
            bin_min = bin_idx * 0.1
            bin_max = (bin_idx + 1) * 0.1
            bin_rows = [r for r in axis_rows if bin_min <= r.get(f"{axis}_score", -1) < bin_max or (bin_idx == 9 and r.get(f"{axis}_score", -1) == 1.0)]
            n_total = len(bin_rows)
            n_break = len([r for r in bin_rows if r.get("label") == 1])
            n_normal = n_total - n_break
            precision = (n_break / n_total) if n_total > 0 else 0.0
            
            # 파단 샘플의 IoU 통계
            break_ious = [r.get(f"{axis}_max_iou") for r in bin_rows if r.get("label") == 1 and r.get(f"{axis}_max_iou") is not None]
            mean_iou = float(np.mean(break_ious)) if break_ious else None
            median_iou = float(np.median(break_ious)) if break_ious else None
            
            bin_stats.append({
                "bin_index": bin_idx,
                "conf_range": f"[{bin_min:.1f}, {bin_max:.1f})",
                "total_samples": n_total,
                "break_samples": n_break,
                "normal_samples": n_normal,
                "precision": round(precision, 4),
                "break_mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
                "break_median_iou": round(median_iou, 4) if median_iou is not None else None,
            })
        
        # threshold별 지표 (0.1, 0.2, ..., 0.9)
        threshold_stats = []
        total_break = len([r for r in axis_rows if r.get("label") == 1])
        for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            above_th = [r for r in axis_rows if r.get(f"{axis}_score", 0) >= th]
            n_above = len(above_th)
            n_break_above = len([r for r in above_th if r.get("label") == 1])
            precision_at_th = (n_break_above / n_above) if n_above > 0 else 0.0
            recall_at_th = (n_break_above / total_break) if total_break > 0 else 0.0
            threshold_stats.append({
                "threshold": th,
                "samples_above_threshold": n_above,
                "break_samples_above": n_break_above,
                "precision": round(precision_at_th, 4),
                "recall": round(recall_at_th, 4),
            })
        
        # 추천 threshold: precision >= 0.5 이면서 recall >= 0.95 를 만족하는 최소 threshold
        recommended_th = None
        for ts in threshold_stats:
            if ts["precision"] >= 0.5 and ts["recall"] >= 0.95:
                recommended_th = ts["threshold"]
                break
        
        conf_bin_stats[axis] = bin_stats
        conf_threshold_stats[axis] = {
            "total_break_samples": total_break,
            "threshold_sweep": threshold_stats,
            "recommended_threshold_precision_0_5_recall_0_95": recommended_th,
        }
    
    # 저장
    conf_stats_path = scope_dir / "confidence_statistics.json"
    with open(conf_stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "bin_statistics": conf_bin_stats,
            "threshold_statistics": conf_threshold_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"저장: {conf_stats_path}")
    
    # 텍스트 요약
    conf_summary_lines = []
    conf_summary_lines.append("=" * 80)
    conf_summary_lines.append("축별 Confidence 기반 파단 판정 기준")
    conf_summary_lines.append("=" * 80)
    conf_summary_lines.append("")
    for axis in AXIS_NAMES:
        if axis not in conf_threshold_stats:
            continue
        rec_th = conf_threshold_stats[axis]["recommended_threshold_precision_0_5_recall_0_95"]
        conf_summary_lines.append(f"[{axis.upper()}축]")
        conf_summary_lines.append(f"  추천 threshold (Precision ≥ 0.5 & Recall ≥ 0.95): {rec_th if rec_th is not None else '해당 없음'}")
        if rec_th is not None:
            th_entry = [t for t in conf_threshold_stats[axis]["threshold_sweep"] if t["threshold"] == rec_th][0]
            conf_summary_lines.append(f"    - Confidence {rec_th} 이상이면 파단 가능성 50% 이상")
            conf_summary_lines.append(f"    - 해당 threshold에서 Recall: {th_entry['recall']:.3f}")
            conf_summary_lines.append(f"    - 파단 샘플 중 {th_entry['recall']*100:.1f}%를 검출 가능 (95% 이상 목표)")
        conf_summary_lines.append("")
    conf_summary_lines.append("=" * 80)
    
    conf_summary_path = scope_dir / "confidence_threshold_recommendation.txt"
    with open(conf_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(conf_summary_lines))
    print(f"저장: {conf_summary_path}")
    print("\n" + "\n".join(conf_summary_lines))
    
    # ============================================================================
    # conf 관련 시각화 (축별 3개 그림)
    # ============================================================================
    for axis in AXIS_NAMES:
        if axis not in conf_bin_stats:
            continue
        
        axis_rows = [r for r in rows if r.get(f"{axis}_has_box")]
        if not axis_rows:
            continue
        
        break_axis_rows = [r for r in axis_rows if r.get("label") == 1]
        normal_axis_rows = [r for r in axis_rows if r.get("label") == 0]
        
        # 1) conf 분포 히스토그램 (정상 vs 파단)
        break_scores = [r.get(f"{axis}_score", 0) for r in break_axis_rows if r.get(f"{axis}_score") is not None]
        normal_scores = [r.get(f"{axis}_score", 0) for r in normal_axis_rows if r.get(f"{axis}_score") is not None]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        if break_scores:
            ax.hist(break_scores, bins=20, alpha=0.7, label=f"Break (n={len(break_scores)})", color="red")
        if normal_scores:
            ax.hist(normal_scores, bins=20, alpha=0.7, label=f"Normal (n={len(normal_scores)})", color="blue")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title(f"Confidence Distribution [{axis.upper()} axis]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        conf_dist_path = scope_dir / f"conf_distribution_{axis}.png"
        plt.savefig(conf_dist_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"저장: {conf_dist_path}")
        
        # 2) conf vs IoU scatter (파단만)
        break_conf_iou = [(r.get(f"{axis}_score"), r.get(f"{axis}_max_iou")) 
                          for r in break_axis_rows 
                          if r.get(f"{axis}_score") is not None and r.get(f"{axis}_max_iou") is not None]
        if break_conf_iou:
            confs, ious = zip(*break_conf_iou)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(confs, ious, alpha=0.6, s=20)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("IoU")
            ax.set_title(f"Confidence vs IoU (Break samples) [{axis.upper()} axis]")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            conf_iou_scatter_path = scope_dir / f"conf_vs_iou_{axis}.png"
            plt.savefig(conf_iou_scatter_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"저장: {conf_iou_scatter_path}")
        
        # 3) threshold sweep: Precision / Recall vs conf threshold
        threshold_sweep = conf_threshold_stats[axis]["threshold_sweep"]
        ths = [t["threshold"] for t in threshold_sweep]
        precs = [t["precision"] for t in threshold_sweep]
        recs = [t["recall"] for t in threshold_sweep]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ths, precs, marker="o", label="Precision", lw=2)
        ax.plot(ths, recs, marker="s", label="Recall", lw=2)
        recommended_th = conf_threshold_stats[axis]["recommended_threshold_precision_0_5_recall_0_95"]
        if recommended_th is not None:
            ax.axvline(recommended_th, color="gray", ls="--", label=f"Recommended th={recommended_th:.1f}")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Precision / Recall vs Confidence Threshold [{axis.upper()} axis]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        threshold_sweep_path = scope_dir / f"conf_threshold_sweep_{axis}.png"
        plt.savefig(threshold_sweep_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"저장: {threshold_sweep_path}")
    
    # conf bin별 이미지 저장 (전체 샘플) - 시간 단축을 위해 옵션으로 건너뛸 수 있음
    if args.skip_sample_images:
        print("옵션 --skip-sample-images: 샘플별 이미지 생성을 건너뜁니다.")
    else:
        # row 에 _idx(샘플 인덱스)가 있으면 그대로 사용. 경로 매칭 없이 인덱스로 접근하면
        # Windows에서 캐시 생성 후 WSL에서 실행할 때 csv_path 형식 차이로 매칭 실패하는 문제를 피할 수 있다.
        # _idx가 없는 경우(기존 run) sample_id로 csv_paths에서 파일명 기준으로 찾는 대체 로직 추가.
        to_draw = [(cat, r) for cat, list_rows in category_lists.items() for r in list_rows]
        drawn_count = 0
        skip_reasons = {"no_idx": 0, "invalid_idx": 0, "not_found": 0, "plot_error": 0}
        for category, r in tqdm(to_draw, desc="conf bin별 이미지 생성"):
            idx = r.get("_idx")
            if idx is None:
                # _idx가 없으면 sample_id로 csv_paths에서 찾기 (파일명 기준 매칭)
                sample_id = r.get("sample_id")
                if not sample_id:
                    skip_reasons["no_idx"] += 1
                    continue
                idx = None
                for j, p in enumerate(csv_paths):
                    # csv_path의 파일명에서 _OUT_processed를 제거한 것이 sample_id와 일치하는지 확인
                    fname = Path(p).stem.replace("_OUT_processed", "")
                    if fname == sample_id:
                        idx = j
                        break
                if idx is None:
                    skip_reasons["not_found"] += 1
                    continue
            if idx < 0 or idx >= n_samples:
                skip_reasons["invalid_idx"] += 1
                continue
            csv_path = csv_paths[idx]
            sample_id = r["sample_id"]
            meta = metas[idx]
            label = labels[idx]
            boxes_by_axis, best_box_by_axis, score_by_axis, top3_by_axis, _, conf_map = _compute_sample_boxes_top3(
                idx, csv_path, meta, label, preds_by_axis, preds_conf_by_axis,
                args.min_box_ratio, args.min_box_degree_span, args.min_box_height_span, has_confidence,
            )
            label_dir = break_dir if label == 1 else normal_dir
            boxes_json_path = label_dir / f"{sample_id}_pred_boxes.json"
            top3_iou_by_axis = None
            if boxes_json_path.exists():
                with open(boxes_json_path, "r", encoding="utf-8") as f:
                    boxes_json = json.load(f)
                top3_iou_by_axis = {}
                for ax in AXIS_NAMES:
                    if ax not in boxes_json.get("axes", {}):
                        continue
                    top3_iou_by_axis[ax] = [entry.get("iou") for entry in boxes_json["axes"][ax]]
            # conf bin 폴더: 해당 축만 강조
            # category 형태: conf_bin_<axis>_<bin_idx> -> 해당 axis의 1순위 박스만 강조
            if category.startswith("conf_bin_"):
                parts = category.split("_")
                if len(parts) == 4 and parts[2] in AXIS_NAMES:
                    axis_from_cat = parts[2]
                    if axis_from_cat in top3_by_axis and top3_by_axis[axis_from_cat]:
                        highlight_rank_by_axis = {axis_from_cat: 0}
                    else:
                        highlight_rank_by_axis = None
                else:
                    highlight_rank_by_axis = None
            else:
                highlight_rank_by_axis = None
            # conf_bin_<axis>_<bin_idx> -> test_result_img/conf_<bin_from>_<bin_to>/<axis>/ 구조로 저장
            if category.startswith("conf_bin_"):
                parts = category.split("_")
                axis_for_dir = parts[2] if len(parts) >= 4 else "x"
                try:
                    bin_idx_for_dir = int(parts[3])
                except (IndexError, ValueError):
                    bin_idx_for_dir = 0
                bin_dir_name = f"conf_{bin_idx_for_dir}_{bin_idx_for_dir + 1}"
                axis_dir = summary_dir / bin_dir_name / axis_for_dir
            else:
                # 예외적으로 다른 카테고리가 생길 경우 이전 방식 유지
                axis_dir = summary_dir / category
            out_path = axis_dir / f"{sample_id}_pred_candidates.png"
            try:
                plot_csv_with_boxes(csv_path, boxes_by_axis, best_box_by_axis, score_by_axis, out_path,
                    conf_by_axis=conf_map, draw_candidates=False, confidence_threshold=args.confidence_threshold, use_confidence=has_confidence, top3_by_axis=top3_by_axis,
                    min_box_ratio=args.min_box_ratio, min_box_degree_span=args.min_box_degree_span, min_box_height_span=args.min_box_height_span,
                    highlight_rank_by_axis=highlight_rank_by_axis, top3_iou_by_axis=top3_iou_by_axis, draw_only_highlighted=True)
                drawn_count += 1
            except Exception as e:
                skip_reasons["plot_error"] += 1
                print(f"경고: 이미지 저장 실패 {out_path}: {e}")
        if to_draw:
            print(f"conf bin별 이미지 생성 완료: {drawn_count}건 (축별 x/y/z, conf 0.1 단위 bin, 총 30개 폴더)")
            if drawn_count == 0:
                print(f"  디버깅: skip 이유 - no_idx={skip_reasons['no_idx']}, not_found={skip_reasons['not_found']}, invalid_idx={skip_reasons['invalid_idx']}, plot_error={skip_reasons['plot_error']}")

    # IoU/conf 분포 및 IoU–conf 연관성 그래프 (전체 + 축별 x, y, z 총 4개)
    for ax in [None, "x", "y", "z"]:
        _plot_eval_distributions(rows, out_dir, has_confidence, axis=ax)

    # F1 / Precision / Recall vs conf threshold 곡선 (축별 + overall)
    _plot_f1_threshold_sweep(rows, out_dir, has_confidence)

    # Precision–Recall 곡선 (축별 + overall)
    _plot_precision_recall_curve(rows, out_dir, has_confidence)

    # 평가 지표 계산 (축별·전체 상관계수, break/normal 분리도 등)
    metrics = _compute_eval_metrics(rows, has_confidence)
    with open(scope_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2차 Hard 모델(conf head) 학습 곡선 이미지 생성
    _plot_training_curves_2nd(second_stage_run_dir, out_dir)

    summary = {
        "timestamp": out_dir.name,
        "has_confidence": has_confidence,
        "confidence_threshold": args.confidence_threshold,
        "num_samples": len(rows),
        "second_stage_run": second_stage_run_dir.name,
        "first_stage_run": first_stage_run_dir.name,
        "metrics": metrics,
    }
    # 2차 학습 설정 요약 포함
    if second_stage_config:
        summary["training_config"] = {
            "data": second_stage_config.get("data", {}),
            "training": second_stage_config.get("training", {}),
            "model": second_stage_config.get("model", {}),
            "loss": second_stage_config.get("loss", {}),
        }
    with open(scope_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    overall = metrics.get("overall", {})
    overall_best_f1 = overall.get("best_f1")
    overall_auc = overall.get("auc_high_iou")
    overall_sep = overall.get("separation")
    has_required = all(v is not None for v in [overall_best_f1, overall_auc, overall_sep])
    if args.target_pass_mode == "f1_only":
        overall_pass = (overall_best_f1 is not None) and (overall_best_f1 >= args.target_overall_best_f1)
    else:
        overall_pass = (
            has_required
            and overall_best_f1 >= args.target_overall_best_f1
            and overall_auc >= args.target_overall_auc
            and overall_sep >= args.target_overall_separation
        )

    weak_axes = sorted(
        AXIS_NAMES,
        key=lambda ax: (
            metrics["by_axis"][ax].get("best_f1") or -1.0,
            metrics["by_axis"][ax].get("auc_high_iou") or -1.0,
            metrics["by_axis"][ax].get("separation") or -9999.0,
        ),
    )
    feedback = {
        "stage": "hard_model_2nd",
        "evaluation_dir": str(scope_dir),
        "second_stage_run": second_stage_run_dir.name,
        "first_stage_run": first_stage_run_dir.name,
        "pass": bool(overall_pass),
        "recommended_retrain": bool(not overall_pass),
        "pass_mode": args.target_pass_mode,
        "criteria": {
            "target_overall_best_f1": float(args.target_overall_best_f1),
            "target_overall_auc": float(args.target_overall_auc),
            "target_overall_separation": float(args.target_overall_separation),
        },
        "actual": {
            "overall_best_f1": overall_best_f1,
            "overall_auc_high_iou": overall_auc,
            "overall_separation": overall_sep,
        },
        "weak_axes_order": weak_axes,
    }
    feedback_path = scope_dir / "training_feedback.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print("저장:", feedback_path)
    print(
        "[feedback] pass={} mode={} overall(best_f1={}, auc={}, separation={})".format(
            overall_pass, args.target_pass_mode, overall_best_f1, overall_auc, overall_sep
        )
    )

    # 콘솔에 축별·전체 상관계수 및 지표 출력
    def _fmt(v):
        return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"

    print("\n--- 평가 지표 (IoU-conf 연관성, F1, break/normal 분리) ---")
    print("축      Pearson r   Spearman ρ  AUC(IoU≥0.5)  best_F1   F1_thr   n_pairs  mean_conf(break) mean_conf(normal) separation")
    for ax in AXIS_NAMES:
        m = metrics["by_axis"][ax]
        print(f"  {ax}     {_fmt(m.get('pearson_r')):>8}   {_fmt(m.get('spearman_r')):>8}    {_fmt(m.get('auc_high_iou')):>8}   {_fmt(m.get('best_f1')):>6}   {_fmt(m.get('best_f1_threshold')):>6}   {m['n_pairs']:>6}   {_fmt(m.get('mean_conf_break')):>14}   {_fmt(m.get('mean_conf_normal')):>14}   {_fmt(m.get('separation'))}")
    mo = metrics["overall"]
    print(f"  all   {_fmt(mo.get('pearson_r')):>8}   {_fmt(mo.get('spearman_r')):>8}    {_fmt(mo.get('auc_high_iou')):>8}   {_fmt(mo.get('best_f1')):>6}   {_fmt(mo.get('best_f1_threshold')):>6}   {mo['n_pairs']:>6}   {_fmt(mo.get('mean_conf_break')):>14}   {_fmt(mo.get('mean_conf_normal')):>14}   {_fmt(mo.get('separation'))}")
    print("  separation = (mean_break - mean_normal) / (std_break + std_normal), 클수록 break/normal conf 분리 좋음")
    print("  AUC = conf로 IoU≥0.5 여부 예측 ROC-AUC. best_F1 = conf threshold 스윕 시 IoU≥0.5 예측 F1 최대값, F1_thr = 해당 threshold")

    pd.DataFrame(rows).to_csv(scope_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 추가: 프로젝트 목적 관점에서 2nd-stage 성능을 요약하는 개요 이미지 생성
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        # 1) 축별 Pearson r (conf vs IoU)
        pearsons = [metrics["by_axis"][ax].get("pearson_r") or 0.0 for ax in AXIS_NAMES]
        axes[0, 0].bar([ax.upper() for ax in AXIS_NAMES], pearsons, color=["C0", "C1", "C2"])
        for x, v in zip([ax.upper() for ax in AXIS_NAMES], pearsons):
            axes[0, 0].text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].set_ylabel("Pearson r (conf vs IoU)")
        axes[0, 0].set_title("Correlation between confidence and IoU (per axis)")
        axes[0, 0].grid(True, axis="y", alpha=0.3)

        # 2) Best F1(IoU≥0.5 vs conf) per axis
        f1s = [metrics["by_axis"][ax].get("best_f1") or 0.0 for ax in AXIS_NAMES]
        axes[0, 1].bar([ax.upper() for ax in AXIS_NAMES], f1s, color=["C0", "C1", "C2"])
        for x, v in zip([ax.upper() for ax in AXIS_NAMES], f1s):
            axes[0, 1].text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].set_ylabel("Best F1 (IoU≥0.5 vs conf)")
        axes[0, 1].set_title("F1: confidence로 high-IoU 박스 구분")
        axes[0, 1].grid(True, axis="y", alpha=0.3)

        # 3) break/normal conf 분포 (mean)
        means_break = [metrics["by_axis"][ax].get("mean_conf_break") or 0.0 for ax in AXIS_NAMES]
        means_normal = [metrics["by_axis"][ax].get("mean_conf_normal") or 0.0 for ax in AXIS_NAMES]
        x_pos = np.arange(len(AXIS_NAMES))
        width = 0.35
        axes[1, 0].bar(x_pos - width / 2, means_normal, width, label="Normal", color="C0")
        axes[1, 0].bar(x_pos + width / 2, means_break, width, label="Break", color="C1")
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([ax.upper() for ax in AXIS_NAMES])
        axes[1, 0].set_ylabel("Mean max conf")
        axes[1, 0].set_title("Confidence separation: Normal vs Break")
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis="y", alpha=0.3)

        # 4) 텍스트 요약 (프로젝트 목적 관점)
        axes[1, 1].axis("off")
        overall = metrics.get("overall", {})
        line_overview = [
            "Hard model 2nd-stage evaluation (confidence head)",
            "",
            f"Second-stage run: {second_stage_run_dir.name}",
            f"First-stage run : {first_stage_run_dir.name}",
            f"Num samples: {len(rows)}",
            "",
            "Goal: confidence should be high when IoU is high (reliable box ranking),",
            "      and higher for break samples than normal samples.",
            "",
            f"Overall Pearson r(conf, IoU): {overall.get('pearson_r')}",
            f"Overall AUC(IoU≥0.5 vs conf): {overall.get('auc_high_iou')}",
            f"Overall best F1(IoU≥0.5 vs conf): {overall.get('best_f1')} (threshold={overall.get('best_f1_threshold')})",
            f"Overall separation (break vs normal conf): {overall.get('separation')}",
        ]
        axes[1, 1].text(0.0, 1.0, "\n".join(line_overview), ha="left", va="top", fontsize=9)

        plt.tight_layout()
        overview_path = scope_dir / "overview_summary.png"
        plt.savefig(overview_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("저장:", overview_path)
    except Exception as e:
        print("overview summary 생성 중 오류:", e)

    # 이미지 설명 (학습 곡선 포함)
    desc_lines = [
        "2nd-stage hard model (conf head) evaluation – image descriptions",
        "",
        "eval_distributions.png / eval_distributions_x|y|z.png: IoU & conf distributions, IoU vs conf scatter per axis.",
        "f1_threshold_sweep.png: Precision / Recall / F1 vs conf threshold (IoU≥0.5 = positive), per axis and overall.",
        "precision_recall_curve.png: Precision–Recall curve (Recall x-axis, Precision y-axis). IoU≥0.5 = positive, conf = score. Per axis and overall, AP(Average Precision) in title.",
        "training_curves.png: 2nd-stage conf head training curves (loss, val_loss, accuracy, val_accuracy) per axis (from 12. hard_models_2nd/<run>/histories.json).",
        "overview_summary.png: Summary of Pearson r(conf vs IoU), best F1(IoU≥0.5 vs conf), break vs normal conf separation.",
        "test_result_img/<category>/*.png: Top/bottom 10 samples per axis (break_iou, break_conf, normal_conf).",
    ]
    desc_path = scope_dir / "image_descriptions.txt"
    try:
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(desc_lines))
        print("저장:", desc_path)
    except Exception as e:
        print(f"경고: 이미지 설명 저장 실패: {e}")

    print("완료:", out_dir)


if __name__ == "__main__":
    main()
