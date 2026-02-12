#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end inference script for real server data.

Flow:
1) Load exported package from step 15 (light/hard2/mlp).
2) Fetch raw measurement data from DB by project or pole.
3) Merge and preprocess CSV files.
4) Run light + hard2 + MLP inference.
5) Save Excel with 2 sheets: by_measure, by_pole.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for p in (str(PROJECT_ROOT), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import poledb as PDB  # noqa: E402

IMG_HEIGHT = 304
DEGREE_GRID = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)
HEIGHT_STEP = 0.1
DEGREE_STEP = 5.0


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


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def get_latest_saved_package(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Package base directory not found: {base_dir}")
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and (d / "package_manifest.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No package with package_manifest.json found in: {base_dir}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def find_saved_model_signature(model: tf.types.experimental.GenericFunction):
    if "serve" in model.signatures:
        return model.signatures["serve"]
    return model.signatures[next(iter(model.signatures.keys()))]


class SavedModelRunner:
    def __init__(self, saved_model_dir: Path):
        self.saved_model_dir = saved_model_dir
        self.obj = tf.saved_model.load(str(saved_model_dir))
        self.fn = find_saved_model_signature(self.obj)

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.fn(tf.convert_to_tensor(x, dtype=tf.float32))
        if isinstance(out, dict):
            return next(iter(out.values())).numpy()
        return out.numpy()


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


def get_project_for_pole(server: str, pole_id: str) -> str:
    PDB.poledb_init(server)
    if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
        raise RuntimeError("DB connection failed")
    q1 = "SELECT groupname FROM tb_anal_state WHERE poleid = %s LIMIT 1"
    r1 = PDB.poledb_conn.do_select_pd(q1, [pole_id])
    if r1 is not None and not r1.empty:
        return str(r1.iloc[0]["groupname"]).strip()
    q2 = "SELECT groupname FROM tb_pole WHERE poleid = %s LIMIT 1"
    r2 = PDB.poledb_conn.do_select_pd(q2, [pole_id])
    if r2 is not None and not r2.empty:
        return str(r2.iloc[0]["groupname"]).strip()
    raise ValueError(f"Project not found for pole: {pole_id}")


def get_pole_ids_for_project(project_name: str, max_poles: int = 0) -> List[str]:
    df = PDB.get_pole_list_a(project_name)
    if df is None or df.empty:
        return []
    col = None
    for c in ("poleid", "pole_id", "POLEID", "poleNo", "poleno"):
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]
    pole_ids = [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]
    pole_ids = sorted(list(dict.fromkeys(pole_ids)))
    if max_poles > 0:
        pole_ids = pole_ids[:max_poles]
    return pole_ids


def get_actual_breakstate(project_name: str, pole_id: str) -> Optional[str]:
    """Get latest actual breakstate from DB analysis result table.

    Returns:
    - "B" for break
    - "N" for normal
    - None if not available
    """
    try:
        if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
            return None
        query = """
            SELECT COALESCE(tar.breakstate, 'N') AS breakstate
            FROM tb_anal_result tar
            INNER JOIN tb_anal_state tas ON tar.poleid = tas.poleid
            WHERE tas.groupname = %s
              AND tar.poleid = %s
              AND tar.analstep IN (1, 2)
            ORDER BY tar.analstep DESC, tar.regdate DESC
            LIMIT 1
        """
        result = PDB.poledb_conn.do_select_pd(query, [project_name, pole_id])
        if result is None or result.empty:
            return None
        state = str(result.iloc[0].get("breakstate", "")).strip().upper()
        return "B" if state == "B" else "N"
    except Exception:
        return None


def breakstate_to_label(state: Optional[str]) -> str:
    if state == "B":
        return "break"
    if state == "N":
        return "normal"
    return "unknown"


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def get_row_value(row: pd.Series, key: str, aliases: List[str]) -> Optional[float]:
    for k in [key] + aliases:
        if k in row.index and pd.notna(row.get(k)):
            return row.get(k)
    return None


def build_meas_info(row: pd.Series, measno: int) -> Dict:
    stdegree = safe_float(get_row_value(row, "stdegree", ["stDegree"]))
    eddegree = safe_float(get_row_value(row, "eddegree", ["edDegree"]))
    stheight = safe_float(get_row_value(row, "stheight", ["stHeight"]))
    edheight = safe_float(get_row_value(row, "edheight", ["edHeight"]))
    sttime = get_row_value(row, "sttime", ["stTime"])
    edtime = get_row_value(row, "endtime", ["edtime", "endTime"])
    return {
        "measno": int(measno),
        "devicetype": "OUT",
        "stdegree": stdegree,
        "eddegree": eddegree,
        "stheight": stheight,
        "edheight": edheight,
        "sttime": str(sttime) if sttime is not None else None,
        "endtime": str(edtime) if edtime is not None else None,
    }


def fetch_pole_raw_data(server: str, project_name: str, pole_id: str, raw_project_dir: Path) -> Dict:
    out = {
        "pole_id": pole_id,
        "project_name": project_name,
        "ok": False,
        "reason": "",
        "pole_raw_dir": "",
        "out_count": 0,
        "saved_count": 0,
    }
    pole_dir = raw_project_dir / pole_id
    pole_dir.mkdir(parents=True, exist_ok=True)
    out["pole_raw_dir"] = str(pole_dir)

    try:
        re_out = PDB.get_meas_result(pole_id, "OUT")
        re_in = PDB.get_meas_result(pole_id, "IN")
    except Exception as e:
        out["reason"] = f"measurement query failed: {e}"
        return out

    if re_out is None or re_out.empty:
        out["reason"] = "OUT measurement data not found"
        return out

    saved = 0
    measurements_info: Dict[str, Dict] = {}
    for i in range(len(re_out)):
        measno = int(re_out.iloc[i]["measno"])
        sttime_raw = str(re_out.iloc[i].get("sttime", ""))
        date_part = sttime_raw.split(" ")[0] if sttime_raw else "unknown"
        try:
            out_x = PDB.get_meas_data(pole_id, measno, "OUT", "x")
            out_y = PDB.get_meas_data(pole_id, measno, "OUT", "y")
            out_z = PDB.get_meas_data(pole_id, measno, "OUT", "z")
        except Exception:
            continue
        if out_x is None or out_x.empty or out_y is None or out_y.empty or out_z is None or out_z.empty:
            continue
        out_x.to_csv(pole_dir / f"{pole_id}_{measno}_{date_part}_OUT_x.csv", index=False)
        out_y.to_csv(pole_dir / f"{pole_id}_{measno}_{date_part}_OUT_y.csv", index=False)
        out_z.to_csv(pole_dir / f"{pole_id}_{measno}_{date_part}_OUT_z.csv", index=False)
        measurements_info[f"OUT_{measno}"] = build_meas_info(re_out.iloc[i], measno)
        saved += 1

    # IN data is not used by this pipeline, but stored for traceability.
    if re_in is not None and not re_in.empty:
        for i in range(len(re_in)):
            measno = int(re_in.iloc[i]["measno"])
            sttime_raw = str(re_in.iloc[i].get("sttime", ""))
            date_part = sttime_raw.split(" ")[0] if sttime_raw else "unknown"
            try:
                in_x = PDB.get_meas_data(pole_id, measno, "IN", "x")
            except Exception:
                in_x = None
            if in_x is not None and not in_x.empty:
                in_x.to_csv(pole_dir / f"{pole_id}_{measno}_{date_part}_IN_x.csv", index=False)

    info = {
        "poleid": pole_id,
        "project_name": project_name,
        "breakstate": "N",
        "breakheight": None,
        "breakdegree": None,
        "measurements": measurements_info,
    }
    (pole_dir / f"{pole_id}_normal_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out["out_count"] = int(len(re_out))
    out["saved_count"] = int(saved)
    out["ok"] = saved > 0
    out["reason"] = "OK" if saved > 0 else "valid OUT x/y/z measurements not found"
    return out


def _circular_degree_diff(a: float, b: float) -> float:
    diff = abs(float(a) - float(b))
    return diff if diff <= 180.0 else (360.0 - diff)


def _build_target_heights(stheight: float, edheight: float) -> np.ndarray:
    hmin = min(float(stheight), float(edheight))
    hmax = max(float(stheight), float(edheight))
    if abs(hmax - hmin) < 1e-9:
        return np.array([hmin], dtype=np.float32)
    values = np.arange(hmin, hmax + HEIGHT_STEP, HEIGHT_STEP, dtype=np.float32)
    values = values[(values >= hmin - 1e-9) & (values <= hmax + 1e-9)]
    return np.unique(np.round(values, 3))


def _build_target_degrees(stdegree: float, eddegree: float) -> np.ndarray:
    st = float(stdegree)
    ed = float(eddegree)
    if ed < st:
        d1 = np.arange(st, 360.0 + DEGREE_STEP, DEGREE_STEP, dtype=np.float32)
        d1 = d1[d1 <= 360.0]
        d2 = np.arange(0.0, ed + DEGREE_STEP, DEGREE_STEP, dtype=np.float32)
        d2 = d2[d2 <= ed + 1e-9]
        return np.unique(np.concatenate([d1, d2]))
    d = np.arange(st, ed + DEGREE_STEP, DEGREE_STEP, dtype=np.float32)
    return d[(d >= st - 1e-9) & (d <= ed + 1e-9)]


def _sensor_angles(stdegree: float, eddegree: float, num_channels: int) -> List[float]:
    st = float(stdegree)
    ed = float(eddegree)
    if ed < st:
        angle_range = (360.0 - st) + ed
    else:
        angle_range = ed - st
    if angle_range < 1.0:
        angle_range = 360.0
        st = 0.0
    angles = []
    for idx in range(num_channels):
        k = idx + 1
        angle = (st + (angle_range * k / (num_channels + 1))) % 360.0
        angles.append(float(angle))
    return angles


def _interpolate_axis_to_grid(
    axis_df: pd.DataFrame,
    meas_info: Dict,
    target_heights: np.ndarray,
    target_degrees: np.ndarray,
) -> Optional[pd.DataFrame]:
    if axis_df is None or axis_df.empty:
        return None

    channel_cols = [c for c in axis_df.columns if c.startswith("ch") and c[2:].isdigit()]
    channel_cols = sorted(channel_cols, key=lambda s: int(s[2:]))
    if len(channel_cols) == 0:
        return None

    stheight = meas_info.get("stheight")
    edheight = meas_info.get("edheight")
    stdegree = meas_info.get("stdegree")
    eddegree = meas_info.get("eddegree")
    if stheight is None or edheight is None or stdegree is None or eddegree is None:
        return None

    n_rows = len(axis_df)
    if n_rows <= 0:
        return None
    rel = np.linspace(0.0, 1.0, n_rows, dtype=np.float32)
    heights = float(stheight) + rel * (float(edheight) - float(stheight))
    angles = _sensor_angles(float(stdegree), float(eddegree), len(channel_cols))

    # Precompute baseline to fill rare missing values.
    all_values = []
    for c in channel_cols:
        vals = axis_df[c].dropna().to_numpy(dtype=np.float32)
        if len(vals) > 0:
            all_values.append(vals)
    baseline = float(np.mean(np.concatenate(all_values))) if all_values else 0.0

    rows: List[Dict[str, float]] = []
    for h in target_heights:
        ch_samples: List[Tuple[float, float]] = []
        for idx, c in enumerate(channel_cols):
            y = axis_df[c].to_numpy(dtype=np.float32)
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() == 0:
                continue
            x_valid = heights[valid_mask]
            y_valid = y[valid_mask]
            order = np.argsort(x_valid)
            x_valid = x_valid[order]
            y_valid = y_valid[order]
            if len(x_valid) == 1:
                val = float(y_valid[0])
            else:
                val = float(np.interp(float(h), x_valid, y_valid))
            ch_samples.append((angles[idx], val))

        if not ch_samples:
            continue

        for d in target_degrees:
            nearest = sorted(ch_samples, key=lambda kv: _circular_degree_diff(float(d), kv[0]))[:2]
            if len(nearest) == 1:
                out_val = float(nearest[0][1])
            else:
                d1 = _circular_degree_diff(float(d), nearest[0][0])
                d2 = _circular_degree_diff(float(d), nearest[1][0])
                total = d1 + d2
                if total > 0:
                    w1 = d2 / total
                    w2 = d1 / total
                    out_val = float(nearest[0][1] * w1 + nearest[1][1] * w2)
                else:
                    out_val = float(nearest[0][1])
            if np.isnan(out_val):
                out_val = baseline
            rows.append(
                {
                    "height": round(float(h), 1),
                    "degree": float(d),
                    "value": out_val,
                }
            )

    if not rows:
        return None
    return pd.DataFrame(rows)


def _find_axis_file_by_measno(pole_dir: Path, measno: int, axis: str) -> Optional[Path]:
    candidates = sorted(pole_dir.glob(f"*OUT*{axis}*.csv"))
    for fp in candidates:
        try:
            df = pd.read_csv(fp, nrows=200)
        except Exception:
            continue
        if "measno" in df.columns:
            vals = pd.to_numeric(df["measno"], errors="coerce").dropna().unique()
            if len(vals) == 1 and int(vals[0]) == int(measno):
                return fp
        if f"_{measno}_" in fp.name:
            return fp
    return None


def process_out_files_for_measno(pole_dir: Path, measno: int, meas_info: Dict, output_dir: Path) -> Optional[Path]:
    poleid = pole_dir.name
    project_name = pole_dir.parent.name
    out_subdir = output_dir / project_name / poleid
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_csv = out_subdir / f"{poleid}_{measno}_OUT_processed.csv"
    if out_csv.exists():
        return out_csv

    x_file = _find_axis_file_by_measno(pole_dir, measno, "x")
    y_file = _find_axis_file_by_measno(pole_dir, measno, "y")
    z_file = _find_axis_file_by_measno(pole_dir, measno, "z")
    if x_file is None or y_file is None or z_file is None:
        return None

    try:
        df_x = pd.read_csv(x_file)
        df_y = pd.read_csv(y_file)
        df_z = pd.read_csv(z_file)
    except Exception:
        return None
    if df_x.empty or df_y.empty or df_z.empty:
        return None

    stheight = meas_info.get("stheight")
    edheight = meas_info.get("edheight")
    stdegree = meas_info.get("stdegree")
    eddegree = meas_info.get("eddegree")
    if stheight is None or edheight is None or stdegree is None or eddegree is None:
        return None

    target_heights = _build_target_heights(float(stheight), float(edheight))
    target_degrees = _build_target_degrees(float(stdegree), float(eddegree))
    if len(target_heights) == 0 or len(target_degrees) == 0:
        return None

    x_grid = _interpolate_axis_to_grid(df_x, meas_info, target_heights, target_degrees)
    y_grid = _interpolate_axis_to_grid(df_y, meas_info, target_heights, target_degrees)
    z_grid = _interpolate_axis_to_grid(df_z, meas_info, target_heights, target_degrees)
    if x_grid is None or y_grid is None or z_grid is None:
        return None

    merged = (
        x_grid.rename(columns={"value": "x_value"})
        .merge(y_grid.rename(columns={"value": "y_value"}), on=["height", "degree"], how="outer")
        .merge(z_grid.rename(columns={"value": "z_value"}), on=["height", "degree"], how="outer")
    )
    merged["devicetype"] = "OUT"
    merged = merged.sort_values(["height", "degree"]).reset_index(drop=True)
    merged = merged[["height", "degree", "x_value", "y_value", "z_value", "devicetype"]]
    valid_mask = ~(merged["x_value"].isna() & merged["y_value"].isna() & merged["z_value"].isna())
    merged = merged[valid_mask].copy()
    if merged.empty:
        return None
    merged.to_csv(out_csv, index=False)
    return out_csv


def process_pole_directory(pole_dir: str, output_dir: str) -> int:
    pole_path = Path(pole_dir)
    output_path = Path(output_dir)
    poleid = pole_path.name
    info_break = pole_path / f"{poleid}_break_info.json"
    info_normal = pole_path / f"{poleid}_normal_info.json"
    info_file = info_break if info_break.exists() else info_normal
    if not info_file.exists():
        return 0

    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
    except Exception:
        return 0
    measurements = info.get("measurements", {})
    if not isinstance(measurements, dict):
        return 0

    processed = 0
    for _, meas_info in measurements.items():
        if not isinstance(meas_info, dict):
            continue
        if str(meas_info.get("devicetype", "")).upper() != "OUT":
            continue
        measno = meas_info.get("measno")
        if measno is None:
            continue
        out_csv = process_out_files_for_measno(pole_path, int(measno), meas_info, output_path)
        if out_csv is not None:
            processed += 1
    return processed


def load_csv_for_infer(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    required = {"height", "degree", "x_value", "y_value", "z_value"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("x_value", "y_value", "z_value"):
        vmin = float(out[col].min())
        vmax = float(out[col].max())
        if vmax > vmin:
            out[col] = (out[col].astype(np.float32) - vmin) / (vmax - vmin)
        else:
            out[col] = 0.0
    return out


def remap_degree(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["degree"] = (np.round(out["degree"] / 5.0) * 5.0).astype(np.float32)
    dmin = float(out["degree"].min())
    dmax = float(out["degree"].max())
    if dmax <= 90.0:
        out["degree"] += 90.0
    elif dmin >= 180.0 and dmax <= 270.0:
        out["degree"] -= 90.0
    elif dmin >= 270.0:
        out["degree"] -= 180.0
    return out


def to_image(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    heights = np.sort(df["height"].unique())

    def _grid(col: str) -> np.ndarray:
        grid = (
            df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean")
            .reindex(index=heights, columns=DEGREE_GRID)
            .to_numpy(dtype=np.float32)
        )
        return np.nan_to_num(grid, nan=0.0)

    img = np.stack([_grid("x_value"), _grid("y_value"), _grid("z_value")], axis=-1).astype(np.float32)
    meta = {
        "grid_shape": list(img.shape),
        "original_length": int(len(df)),
        "unique_heights": int(len(heights)),
        "unique_degrees": int(len(DEGREE_GRID)),
    }
    return img, meta


def resize_img_height(img: np.ndarray, target_h: int = IMG_HEIGHT) -> np.ndarray:
    h = int(img.shape[0])
    if h == target_h:
        return img.astype(np.float32, copy=False)
    resized = tf.image.resize(img, size=(target_h, int(img.shape[1])), method="bilinear")
    return resized.numpy().astype(np.float32)


def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = "height",
    feature_min_max: Optional[Dict] = None,
    max_height: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict]]:
    _ = feature_min_max, max_height
    df = load_csv_for_infer(Path(csv_path))
    if df is None:
        return None
    if sort_by == "degree":
        df = df.sort_values(["degree", "height"]).reset_index(drop=True)
    else:
        df = df.sort_values(["height", "degree"]).reset_index(drop=True)
    df = normalize_values(df)
    df = remap_degree(df)
    return to_image(df)


def extract_pole_and_measno(csv_path: Path) -> Tuple[str, str]:
    name = csv_path.name
    m = re.match(r"(.+?)_(\d+)_OUT_processed\.csv$", name)
    if m:
        return m.group(1), m.group(2)
    stem = csv_path.stem.replace("_OUT_processed", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return stem, ""


def decide_label(prob: float, suspect_th: float, break_th: float) -> Tuple[int, str]:
    if prob >= break_th:
        return 2, "break"
    if prob >= suspect_th:
        return 1, "suspect"
    return 0, "normal"


def summarize_by_pole(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "server",
                "project_name",
                "pole_id",
                "actual_breakstate",
                "actual_label",
                "measurement_count",
                "normal_count",
                "suspect_count",
                "break_count",
                "max_mlp_prob",
                "mean_mlp_prob",
                "pole_final_label",
                "is_break_binary_correct",
            ]
        )

    rows = []
    for pole_id, g in df.groupby("pole_id"):
        break_count = int((g["pred_label_num"] == 2).sum())
        suspect_count = int((g["pred_label_num"] == 1).sum())
        normal_count = int((g["pred_label_num"] == 0).sum())
        if break_count > 0:
            pole_label = "break"
        elif suspect_count > 0:
            pole_label = "suspect"
        else:
            pole_label = "normal"

        actual_state = g["actual_breakstate"].iloc[0] if "actual_breakstate" in g.columns else None
        actual_label = breakstate_to_label(actual_state if pd.notna(actual_state) else None)
        pred_break_binary = 1 if break_count > 0 else 0
        if actual_state in ("B", "N"):
            actual_break_binary = 1 if actual_state == "B" else 0
            is_break_binary_correct = int(pred_break_binary == actual_break_binary)
        else:
            is_break_binary_correct = np.nan

        rows.append(
            {
                "server": g["server"].iloc[0],
                "project_name": g["project_name"].iloc[0],
                "pole_id": pole_id,
                "actual_breakstate": actual_state if pd.notna(actual_state) else "UNK",
                "actual_label": actual_label,
                "measurement_count": int(len(g)),
                "normal_count": normal_count,
                "suspect_count": suspect_count,
                "break_count": break_count,
                "max_mlp_prob": float(g["mlp_prob_break"].max()),
                "mean_mlp_prob": float(g["mlp_prob_break"].mean()),
                "pole_final_label": pole_label,
                "is_break_binary_correct": is_break_binary_correct,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["break_count", "suspect_count", "max_mlp_prob"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end inference by project or pole and export Excel.")
    parser.add_argument("--server", required=True, help="server name: main/is/kh")
    parser.add_argument("--project", default=None, help="project name")
    parser.add_argument("--pole-id", default=None, help="single pole id")
    parser.add_argument("--max-poles", type=int, default=0, help="max poles in project mode (0=all)")
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--saved-package", default=None, help="package name under 15. make_save_model (default: latest)")
    parser.add_argument("--saved-base-dir", default="15. make_save_model", help="saved package base directory")
    parser.add_argument("--output-dir", default="16. test_ai", help="output base directory")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="TensorFlow runtime device")
    args = parser.parse_args()

    if not args.project and not args.pole_id:
        raise ValueError("Specify either --project or --pole-id")

    configure_tf_runtime(args.device)

    # 0) Load package
    saved_base = CURRENT_DIR / args.saved_base_dir
    package_dir = saved_base / args.saved_package if args.saved_package else get_latest_saved_package(saved_base)
    if not package_dir.exists():
        raise FileNotFoundError(f"Model package directory not found: {package_dir}")
    manifest = load_json(package_dir / "package_manifest.json")

    light_saved = package_dir / "light_model" / "saved_model"
    hard_saved = package_dir / "hard_model_2nd" / "saved_model"
    mlp_model_path = package_dir / "mlp_model" / "model" / "mlp_pipeline.joblib"
    mlp_summary_path = package_dir / "mlp_model" / "test_results" / "mlp_summary.json"

    required_paths = [
        light_saved,
        hard_saved / "conf_x",
        hard_saved / "conf_y",
        hard_saved / "conf_z",
        mlp_model_path,
        mlp_summary_path,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Required package path not found: {p}")

    print(f"[PACKAGE] {package_dir}")
    light_runner = SavedModelRunner(light_saved)
    conf_x_runner = SavedModelRunner(hard_saved / "conf_x")
    conf_y_runner = SavedModelRunner(hard_saved / "conf_y")
    conf_z_runner = SavedModelRunner(hard_saved / "conf_z")
    mlp_model = joblib.load(mlp_model_path)
    mlp_summary = load_json(mlp_summary_path)

    weights = mlp_summary.get("weights", {})
    axis_wx = float(weights.get("axis_x", 0.34))
    axis_wy = float(weights.get("axis_y", 0.33))
    axis_wz = float(weights.get("axis_z", 0.33))
    wl = float(weights.get("light", 0.55))
    wh = float(weights.get("hard", 0.45))
    thresholds = mlp_summary.get("thresholds", {})
    suspect_th = float(thresholds.get("suspect_threshold", 0.5))
    break_th = float(thresholds.get("break_threshold", 0.9))

    # 1) Resolve target poles
    PDB.poledb_init(args.server)
    if args.pole_id:
        project_name = args.project or get_project_for_pole(args.server, args.pole_id)
        pole_ids = [args.pole_id]
    else:
        project_name = args.project
        pole_ids = get_pole_ids_for_project(project_name, max_poles=args.max_poles)
        if not pole_ids:
            raise RuntimeError(f"No poles found for project: {project_name}")
    print(f"[TARGET] server={args.server}, project={project_name}, pole_count={len(pole_ids)}")

    # 2) Prepare run directories
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = args.pole_id if args.pole_id else project_name
    safe_target = re.sub(r"[\\/:*?\"<>| ]", "_", str(target_name))
    run_name = f"{ts}_{args.server}_{safe_target}"
    out_base = CURRENT_DIR / args.output_dir
    run_dir = out_base / run_name
    raw_dir = run_dir / "3. raw_pole_data" / "normal" / project_name
    merge_dir = run_dir / "4. merge_data" / "normal"
    raw_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    # 3) Standalone preprocessing uses in-file functions.
    target_h = IMG_HEIGHT

    # 4) Fetch from server -> raw -> merge
    fetch_rows = []
    csv_rows: List[Dict] = []
    print("[FETCH] start")
    for pole_id in tqdm(pole_ids, desc="poles", unit="pole"):
        actual_state = get_actual_breakstate(project_name, pole_id)
        actual_label = breakstate_to_label(actual_state)
        info = fetch_pole_raw_data(args.server, project_name, pole_id, raw_dir)
        info["actual_breakstate"] = actual_state if actual_state is not None else "UNK"
        info["actual_label"] = actual_label
        fetch_rows.append(info)
        if not info["ok"]:
            continue

        pole_raw_dir = Path(info["pole_raw_dir"])
        _ = process_pole_directory(str(pole_raw_dir), str(merge_dir))
        pole_processed_dir = merge_dir / project_name / pole_id
        for p in sorted(pole_processed_dir.glob("*_OUT_processed.csv")):
            pid, measno = extract_pole_and_measno(p)
            csv_rows.append(
                {
                    "server": args.server,
                    "project_name": project_name,
                    "pole_id": pid,
                    "measurement_no": measno,
                    "actual_breakstate": actual_state if actual_state is not None else "UNK",
                    "actual_label": actual_label,
                    "processed_csv_path": str(p),
                }
            )

    if not csv_rows:
        raise RuntimeError("No processed CSV files available after fetch/merge.")

    # 5) Per-measure inference
    rows = []
    images = []
    for rec in csv_rows:
        csv_path = rec["processed_csv_path"]
        row = dict(rec)
        try:
            prep_result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by="height",
                feature_min_max=None,
            )
            if prep_result is None:
                row["status"] = "skip_prepare_failed"
                rows.append(row)
                continue

            img, meta = prep_result
            npts = int(meta.get("original_length", 0))
            row["point_count"] = npts
            if npts < args.min_points or npts > args.max_points:
                row["status"] = f"skip_points_out_of_range({npts})"
                rows.append(row)
                continue

            img = resize_img_height(img, target_h=target_h).astype(np.float32)
            images.append(img)
            row["status"] = "ok"
            rows.append(row)
        except Exception as e:
            row["status"] = f"skip_exception({e})"
            rows.append(row)

    if not images:
        raise RuntimeError("No valid samples for inference. Check point-range or CSV preprocessing.")

    X = np.stack(images, axis=0)
    light_prob = flatten_binary_output(light_runner.predict(X))
    conf_x = flatten_axis_conf(conf_x_runner.predict(X))
    conf_y = flatten_axis_conf(conf_y_runner.predict(X))
    conf_z = flatten_axis_conf(conf_z_runner.predict(X))

    axis_sum = axis_wx + axis_wy + axis_wz
    if axis_sum <= 0:
        axis_sum = 1.0
    wx, wy, wz = axis_wx / axis_sum, axis_wy / axis_sum, axis_wz / axis_sum
    hard_score = (wx * conf_x) + (wy * conf_y) + (wz * conf_z)
    fused_score = (wl * light_prob) + (wh * hard_score)
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
    mlp_prob = mlp_model.predict_proba(features)[:, 1]

    # Fill scores into rows in the same order as valid samples.
    valid_pos = 0
    for r in rows:
        actual_state = r.get("actual_breakstate")
        if actual_state == "B":
            actual_num = 1
        elif actual_state == "N":
            actual_num = 0
        else:
            actual_num = np.nan
        r["actual_label_num"] = actual_num

        if r.get("status") != "ok":
            r["light_prob"] = np.nan
            r["conf_x"] = np.nan
            r["conf_y"] = np.nan
            r["conf_z"] = np.nan
            r["hard_score"] = np.nan
            r["fused_score"] = np.nan
            r["mlp_prob_break"] = np.nan
            r["pred_label_num"] = np.nan
            r["pred_label"] = "skip"
            r["is_break_binary_correct"] = np.nan
            continue

        lp = float(light_prob[valid_pos])
        sx = float(conf_x[valid_pos])
        sy = float(conf_y[valid_pos])
        sz = float(conf_z[valid_pos])
        hs = float(hard_score[valid_pos])
        fs = float(fused_score[valid_pos])
        mp = float(mlp_prob[valid_pos])
        label_num, label_name = decide_label(mp, suspect_th, break_th)

        r["light_prob"] = lp
        r["conf_x"] = sx
        r["conf_y"] = sy
        r["conf_z"] = sz
        r["hard_score"] = hs
        r["fused_score"] = fs
        r["mlp_prob_break"] = mp
        r["pred_label_num"] = label_num
        r["pred_label"] = label_name

        if actual_state in ("B", "N"):
            actual_break_binary = 1 if actual_state == "B" else 0
            pred_break_binary = 1 if label_num == 2 else 0
            r["is_break_binary_correct"] = int(actual_break_binary == pred_break_binary)
        else:
            r["is_break_binary_correct"] = np.nan
        valid_pos += 1

    df_measure = pd.DataFrame(rows)
    if "measurement_no" in df_measure.columns:
        df_measure["_measurement_no_num"] = pd.to_numeric(df_measure["measurement_no"], errors="coerce")
        df_measure = df_measure.sort_values(["pole_id", "_measurement_no_num", "measurement_no"], ascending=[True, True, True]).drop(columns=["_measurement_no_num"])
    else:
        df_measure = df_measure.sort_values(["pole_id"]).copy()
    df_measure = df_measure.reset_index(drop=True)
    df_measure_ok = df_measure[df_measure["status"] == "ok"].copy()
    df_pole = summarize_by_pole(df_measure_ok)

    # 6) Save outputs (2-sheet Excel + summary)
    excel_path = run_dir / f"prediction_{args.server}_{safe_target}_{ts}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_measure.to_excel(writer, index=False, sheet_name="by_measure")
        df_pole.to_excel(writer, index=False, sheet_name="by_pole")

    run_summary = {
        "run_name": run_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "server": args.server,
        "project_name": project_name,
        "input_mode": "pole" if args.pole_id else "project",
        "pole_count": len(pole_ids),
        "meas_total": int(len(df_measure)),
        "meas_inferred": int(len(df_measure_ok)),
        "meas_skipped": int(len(df_measure) - len(df_measure_ok)),
        "actual_distribution_by_pole": {
            "break": int((df_pole["actual_breakstate"] == "B").sum()) if not df_pole.empty else 0,
            "normal": int((df_pole["actual_breakstate"] == "N").sum()) if not df_pole.empty else 0,
            "unknown": int((df_pole["actual_breakstate"] == "UNK").sum()) if not df_pole.empty else 0,
        },
        "thresholds": {
            "suspect_threshold": suspect_th,
            "break_threshold": break_th,
        },
        "weights": {
            "light": wl,
            "hard": wh,
            "axis_x": wx,
            "axis_y": wy,
            "axis_z": wz,
        },
        "model_package": str(package_dir),
        "excel_path": str(excel_path),
        "package_manifest": manifest,
    }
    save_json(run_dir / "run_summary.json", run_summary)
    pd.DataFrame(fetch_rows).to_csv(run_dir / "fetch_status.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"Done: {excel_path}")
    print("- by_measure sheet: per measurement prediction")
    print("- by_pole sheet: per pole summary")
    print(f"run_summary: {run_dir / 'run_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
