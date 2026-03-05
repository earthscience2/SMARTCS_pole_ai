#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라이트 모델 학습용 NPY 데이터를 생성한다."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
# 4. merge_data, 5. edit_data는 main/1. make_data set/ 아래에 있음
DATA_SET_DIR = CURRENT_DIR.parent / "1. make_data set"
IMG_HEIGHT = 304
DEGREE_GRID = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)
# Backward compatibility for modules importing old names.
current_dir = str(CURRENT_DIR)
img_height = IMG_HEIGHT


def load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """CSV를 읽고 필수 컬럼이 있으면 반환한다."""
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
    """x/y/z 값을 파일 단위 min-max 정규화한다."""
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
    """
    degree를 90~180 구간으로 맞춘다.
    - 0~90 -> +90
    - 180~270 -> -90
    - 270~360 -> -180
    """
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
    """전처리된 DataFrame을 (H, 19, 3) 이미지로 변환한다."""
    heights = np.sort(df["height"].unique())

    def grid_for(col: str) -> np.ndarray:
        grid = (
            df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean")
            .reindex(index=heights, columns=DEGREE_GRID)
            .to_numpy(dtype=np.float32)
        )
        return np.nan_to_num(grid, nan=0.0)

    x_grid = grid_for("x_value")
    y_grid = grid_for("y_value")
    z_grid = grid_for("z_value")
    img = np.stack([x_grid, y_grid, z_grid], axis=-1).astype(np.float32)
    meta = {
        "grid_shape": list(img.shape),
        "original_length": int(len(df)),
        "unique_heights": int(len(heights)),
        "unique_degrees": int(len(DEGREE_GRID)),
    }
    return img, meta


def resize_height(img: np.ndarray, target_h: int = IMG_HEIGHT) -> np.ndarray:
    """높이 축만 리사이즈한다."""
    h, _, _ = img.shape
    if h == target_h:
        return img
    return zoom(img, (target_h / h, 1.0, 1.0), order=1).astype(np.float32)


def resize_img_height(img: np.ndarray, target_h: int = IMG_HEIGHT) -> np.ndarray:
    """기존 코드 호환 alias."""
    return resize_height(img, target_h=target_h)


def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = "height",
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    max_height: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    기존 코드 호환 함수.
    `feature_min_max`, `max_height`는 현재 로직에서 사용하지 않는다.
    """
    _ = feature_min_max, max_height
    df = load_csv(Path(csv_path))
    if df is None:
        return None

    if sort_by == "degree":
        df = df.sort_values(["degree", "height"]).reset_index(drop=True)
    else:
        df = df.sort_values(["height", "degree"]).reset_index(drop=True)

    df = normalize_values(df)
    df = remap_degree(df)
    img, meta = to_image(df)
    return img, meta


def collect_break_files_from_edit_data(
    edit_data_dir: str = "5. edit_data",
    merge_data_dir: str = "4. merge_data",
) -> List[Tuple[str, str, str, int]]:
    """ROI에서 삭제되지 않은 파단 CSV 목록을 수집한다."""
    edit_break = DATA_SET_DIR / edit_data_dir / "break"
    merge_break = DATA_SET_DIR / merge_data_dir / "break"
    result: List[Tuple[str, str, str, int]] = []
    seen: set[str] = set()

    if not edit_break.exists():
        print(f"경고: 디렉터리가 없습니다. {edit_break}")
        return result

    for project_dir in edit_break.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            poleid = pole_dir.name
            for roi_path in pole_dir.glob("*roi_info*.json"):
                try:
                    with roi_path.open("r", encoding="utf-8") as f:
                        info = json.load(f)
                    if info.get("deleted") is True:
                        continue
                except Exception:
                    continue

                stem = roi_path.stem
                csv_name = stem[:-9] + ".csv" if stem.endswith("_roi_info") else stem.replace("_roi_info", "") + ".csv"
                csv_edit = pole_dir / csv_name
                csv_merge = merge_break / project_name / poleid / csv_name
                csv_path = csv_edit if csv_edit.exists() else (csv_merge if csv_merge.exists() else None)
                if csv_path is None:
                    continue
                key = str(csv_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                result.append((str(csv_path), project_name, poleid, 1))
    return result


def collect_all_crop_files(data_dir: str, is_break: bool) -> List[Tuple[str, str, str, int]]:
    """merge_data에서 break/normal CSV를 수집한다."""
    base = DATA_SET_DIR / data_dir / ("break" if is_break else "normal")
    result: List[Tuple[str, str, str, int]] = []
    if not base.exists():
        print(f"경고: 디렉터리가 없습니다. {base}")
        return result

    patterns = ["*_OUT_processed.csv"] if is_break else ["*_OUT_processed.csv", "*_normal_processed.csv"]
    label = 1 if is_break else 0
    for project_dir in base.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            poleid = pole_dir.name
            for pattern in patterns:
                for csv_path in pole_dir.glob(pattern):
                    result.append((str(csv_path), project_name, poleid, label))
    return result


def get_latest_run_dir(base_dir: str = "1. light_train_data") -> Optional[Path]:
    """최신 run 디렉터리를 반환한다."""
    base = CURRENT_DIR / base_dir
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and len(d.name) == 13 and d.name[8] == "_"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def process_cropped_data(
    data_dir: str = "4. merge_data",
    output_dir: str = "1. light_train_data",
    break_data_dir: str = "5. edit_data",
    sort_by: str = "height",
    min_points: int = 200,
    max_points: int = 400,
    run_subdir: Optional[str] = None,
) -> Optional[Path]:
    """CSV를 읽어 학습/테스트 NPY를 생성한다."""
    output_base = CURRENT_DIR / output_dir
    output_base.mkdir(parents=True, exist_ok=True)
    run_name = run_subdir or datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 경로: {run_dir}")

    break_files = collect_break_files_from_edit_data(edit_data_dir=break_data_dir, merge_data_dir=data_dir)
    normal_files = collect_all_crop_files(data_dir=data_dir, is_break=False)
    if not break_files:
        raise FileNotFoundError("파단 데이터가 없습니다. 5. edit_data 또는 4. merge_data/break를 확인하세요.")
    if not normal_files:
        raise FileNotFoundError("정상 데이터가 없습니다. 4. merge_data/normal을 확인하세요.")

    print(f"파단 CSV: {len(break_files)}개, 정상 CSV: {len(normal_files)}개")

    images: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict] = []
    failed_files: List[Dict] = []

    def process_one(csv_path: str, project_name: str, poleid: str, label: int, normal_mode: bool) -> bool:
        result = prepare_sequence_from_csv(csv_path=csv_path, sort_by=sort_by)
        if result is None:
            failed_files.append({"csv_path": csv_path, "project_name": project_name, "poleid": poleid, "reason": "CSV 로드 실패"})
            return False

        img, meta = result
        point_count = int(meta["original_length"])
        if point_count < min_points:
            failed_files.append(
                {"csv_path": csv_path, "project_name": project_name, "poleid": poleid, "reason": f"포인트 부족({point_count})"}
            )
            return False
        if normal_mode:
            if point_count >= max_points:
                failed_files.append(
                    {"csv_path": csv_path, "project_name": project_name, "poleid": poleid, "reason": f"정상 포인트 초과({point_count})"}
                )
                return False
        elif point_count > max_points:
            failed_files.append(
                {"csv_path": csv_path, "project_name": project_name, "poleid": poleid, "reason": f"파단 포인트 초과({point_count})"}
            )
            return False

        img = resize_height(img, target_h=IMG_HEIGHT)
        images.append(img)
        labels.append(label)
        metadata.append(
            {
                **meta,
                "csv_path": csv_path,
                "project_name": project_name,
                "poleid": poleid,
                "label": int(label),
                "img_height": IMG_HEIGHT,
            }
        )
        return True

    for csv_path, project_name, poleid, label in tqdm(break_files, desc="파단 처리", unit="file"):
        process_one(csv_path, project_name, poleid, label, normal_mode=False)

    break_count = int(sum(1 for l in labels if l == 1))
    max_normal_samples = break_count * 10
    normal_kept = 0
    for csv_path, project_name, poleid, label in tqdm(normal_files, desc="정상 처리", unit="file"):
        if normal_kept >= max_normal_samples:
            break
        kept = process_one(csv_path, project_name, poleid, label, normal_mode=True)
        if kept:
            normal_kept += 1

    if not images:
        print("생성된 샘플이 없습니다.")
        return None

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    y_flat = y.ravel()
    stratify = y_flat if len(np.unique(y_flat)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=stratify
    )

    train_dir = run_dir / "train"
    test_dir = run_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "break_imgs_train.npy", X_train)
    np.save(train_dir / "break_labels_train.npy", y_train)
    np.save(test_dir / "break_imgs_test.npy", X_test)
    np.save(test_dir / "break_labels_test.npy", y_test)
    np.save(run_dir / "break_imgs.npy", X)
    np.save(run_dir / "break_labels.npy", y)

    meta_path = run_dir / "break_imgs_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_subdir": run_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "total_samples": int(len(metadata)),
                "img_height": IMG_HEIGHT,
                "sort_by": sort_by,
                "min_points": int(min_points),
                "max_points": int(max_points),
                "normal_max_samples": int(max_normal_samples),
                "data_shape": list(X.shape),
                "samples": metadata,
                "failed_files": failed_files,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("저장 완료")
    print(f"  train X: {train_dir / 'break_imgs_train.npy'}")
    print(f"  train y: {train_dir / 'break_labels_train.npy'}")
    print(f"  test X : {test_dir / 'break_imgs_test.npy'}")
    print(f"  test y : {test_dir / 'break_labels_test.npy'}")
    print(f"  metadata: {meta_path}")
    print(f"  class 분포: {dict(zip(*np.unique(y, return_counts=True)))}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라이트 모델 학습용 NPY 생성")
    parser.add_argument("--data-dir", default="4. merge_data", help="merge_data 기준 디렉터리")
    parser.add_argument("--output-dir", default="1. light_train_data", help="출력 디렉터리")
    parser.add_argument("--break-data-dir", default="5. edit_data", help="ROI 기반 파단 데이터 디렉터리")
    parser.add_argument("--sort-by", default="height", choices=["height", "degree"])
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--run-subdir", default=None, help="출력 run 폴더명 (미지정 시 timestamp)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_cropped_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        break_data_dir=args.break_data_dir,
        sort_by=args.sort_by,
        min_points=args.min_points,
        max_points=args.max_points,
        run_subdir=args.run_subdir,
    )


if __name__ == "__main__":
    main()
