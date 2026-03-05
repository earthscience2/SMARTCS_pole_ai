"""?쒗???숈뒿 ?곗씠??以鍮?(CSV ??NPY) - Hard 紐⑤뜽??bbox ?ы븿 ?숈뒿 ?곗씠???앹꽦"""

import os
import re
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# ?ㅽ겕由쏀듃 ?붾젆?좊━ 湲곗? 寃쎈줈 (?ㅽ뻾 ?꾩튂? 臾닿?)
current_dir = os.path.dirname(os.path.abspath(__file__))
img_height = 304

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _resolve_existing_dir(path_str: str, candidates: List[Path]) -> Optional[Path]:
    p = Path(path_str)
    if p.is_absolute():
        return p if p.exists() else None
    for base in candidates:
        cand = base / p
        if cand.exists():
            return cand
    return None


def _resolve_data_root_dir(data_dir: str) -> Path:
    base_candidates = [
        Path(current_dir),
        Path(current_dir).parent / "1. make_data set",
        Path.cwd() / "main" / "1. make_data set",
        Path.cwd(),
    ]
    resolved = _resolve_existing_dir(data_dir, base_candidates)
    if resolved is None:
        if Path(data_dir).is_absolute():
            return Path(data_dir)
        return Path(current_dir) / data_dir
    return resolved


def _resolve_edit_data_dir(edit_data_dir: str) -> Path:
    base_candidates = [
        Path(current_dir),
        Path(current_dir).parent / "1. make_data set",
        Path.cwd() / "main" / "1. make_data set",
        Path.cwd(),
    ]
    resolved = _resolve_existing_dir(edit_data_dir, base_candidates)
    if resolved is None:
        if Path(edit_data_dir).is_absolute():
            return Path(edit_data_dir)
        return Path(current_dir) / edit_data_dir
    return resolved

# ============================================================================
# 1) CSV 濡쒕뱶 / ?쒗???대?吏) ?앹꽦 ?좏떥
# ============================================================================


def load_crop_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """?щ∼??CSV ?뚯씪 濡쒕뱶."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print("[ERR] read_csv failed:", csv_path)
        print("   err =", repr(e))
        return None


def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = "height",
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    max_height: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """CSV ?뚯씪?먯꽌 ?쒗???대?吏) ?곗씠???앹꽦 (height, degree, x, y, z ?ы븿, 媛곴컖 0~1 ?뺢퇋??."""
    df = load_crop_csv(csv_path)
    if df is None:
        return None

    required_cols = ["height", "degree", "x_value", "y_value", "z_value"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None

    if sort_by == "height":
        df = df.sort_values(["height", "degree"]).reset_index(drop=True)
    elif sort_by == "degree":
        df = df.sort_values(["degree", "height"]).reset_index(drop=True)
    else:
        df = df.sort_values(["height", "degree"]).reset_index(drop=True)

    if feature_min_max is None:
        feature_min_max = {
            "height": (df["height"].min(), df["height"].max()),
            "degree": (df["degree"].min(), df["degree"].max()),
            "x_value": (df["x_value"].min(), df["x_value"].max()),
            "y_value": (df["y_value"].min(), df["y_value"].max()),
            "z_value": (df["z_value"].min(), df["z_value"].max()),
        }

    value_cols = ["x_value", "y_value", "z_value"]
    for col in value_cols:
        vmin, vmax = feature_min_max[col]
        if vmax > vmin:
            df[col] = (df[col].astype(np.float32) - vmin) / (vmax - vmin)
        else:
            df[col] = 0.0

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

    def make_grid(col: str) -> np.ndarray:
        g = (
            df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean")
            .reindex(index=heights, columns=degrees)
            .to_numpy(dtype=np.float32)
        )
        return np.nan_to_num(g, nan=0.0)

    x_grid = make_grid("x_value")
    y_grid = make_grid("y_value")
    z_grid = make_grid("z_value")
    img = np.stack([x_grid, y_grid, z_grid], axis=-1).astype(np.float32)

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


# ============================================================================
# 2) ?뚯씪 ?섏쭛 / ROI bbox ?좏떥
# ============================================================================


def collect_all_crop_files(data_dir: str, is_break: bool) -> List[Tuple[str, str, str, int]]:
    """
    data_dir/{break|normal} ?꾨옒?먯꽌 *_OUT_processed.csv ?뚯씪 ?섏쭛.
    諛섑솚: (csv_path, project_name, poleid, label)
    """
    base_dir = _resolve_data_root_dir(data_dir)
    base_dir = base_dir / ("break" if is_break else "normal")
    out = []

    if not base_dir.exists():
        raise FileNotFoundError(
            f"CSV ?섏쭛 寃쎈줈瑜?李얠? 紐삵뻽?듬땲?? {base_dir}\n"
            f"?낅젰 data_dir='{data_dir}'\n"
            f"沅뚯옣: --data-dir \"4. merge_data\" (main/1. make_data set ?섏쐞 ?먮룞 ?먯깋)"
        )

    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name

        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            poleid = pole_dir.name
            for csv_file in pole_dir.glob("*_OUT_processed.csv"):
                out.append((str(csv_file), project_name, poleid, 1 if is_break else 0))
    return out


def parse_roi_bbox(roi_info: dict, k: int) -> List[List[float]]:
    """roi_{k}_regions?먯꽌 [hc, hw, dc, dw] 由ъ뒪??諛섑솚."""
    out = []
    regions = roi_info.get(f"roi_{k}_regions")
    if not isinstance(regions, list):
        return out

    for r in regions:
        hmin = r.get("height_min")
        hmax = r.get("height_max")
        dmin = r.get("degree_min")
        dmax = r.get("degree_max")
        if None in (hmin, hmax, dmin, dmax):
            continue
        try:
            hmin, hmax, dmin, dmax = map(float, (hmin, hmax, dmin, dmax))
        except Exception:
            continue
        hc = (hmin + hmax) / 2.0
        hw = hmax - hmin
        dc = (dmin + dmax) / 2.0
        dw = dmax - dmin
        out.append([hc, hw, dc, dw])
    return out


def get_sample_id_from_csv(csv_path: str) -> Optional[str]:
    """0621R481_2_OUT_processed.csv -> 0621R481_2"""
    name = Path(csv_path).name
    m = re.match(r"(.+)_OUT_processed\.csv$", name)
    return m.group(1) if m else None


def match_roi_json_from_csv(csv_path: str, edit_data_dir: str = "5. edit_data") -> Optional[str]:
    """CSV 寃쎈줈????묓븯??edit_data_dir/break ??roi_info.json 寃쎈줈 諛섑솚."""
    p = Path(csv_path)
    sample_id = get_sample_id_from_csv(csv_path)
    if sample_id is None:
        return None
    project = p.parents[1].name
    poleid = p.parents[0].name
    roi_json = (
        _resolve_edit_data_dir(edit_data_dir)
        / "break"
        / project
        / poleid
        / f"{sample_id}_OUT_processed_roi_info.json"
    )
    return str(roi_json) if roi_json.exists() else None


def load_roi_info_json(json_path: str) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def expand_rois_from_roi_info(roi_info: Optional[dict]) -> List[Tuple[int, List[float]]]:
    """(roi_idx, [hc, hw, dc, dw]) 由ъ뒪??諛섑솚."""
    if roi_info is None:
        return []
    out = []
    for k in (0, 1, 2):
        for b in parse_roi_bbox(roi_info, k):
            out.append((k, b))
    return out


def normalize_bbox_center_width(bbox: List[float], feature_min_max: dict) -> Optional[List[float]]:
    """
    bbox: [hc, hw, dc, dw] (?먮낯 ?ㅼ???
    feature_min_max: metadata['feature_min_max'] with keys 'height','degree'
    return: [hc_n, hw_n, dc_n, dw_n] in [0,1]
    """
    hc, hw, dc, dw = bbox[0], bbox[1], bbox[2], bbox[3]
    h_min, h_max = feature_min_max["height"]
    d_min, d_max = feature_min_max["degree"]
    h_rng = h_max - h_min
    d_rng = d_max - d_min
    if h_rng <= 0 or d_rng <= 0:
        return None
    hc_n = (hc - h_min) / h_rng
    hw_n = hw / h_rng
    dc_n = (dc - d_min) / d_rng
    dw_n = dw / d_rng
    hc_n = float(np.clip(hc_n, 0.0, 1.0))
    dc_n = float(np.clip(dc_n, 0.0, 1.0))
    hw_n = float(np.clip(hw_n, 0.0, 1.0))
    dw_n = float(np.clip(dw_n, 0.0, 1.0))
    return [hc_n, hw_n, dc_n, dw_n]


def _check_sync(stage: str, csv_path: str, imgs: list, labels: list, metadata_list: list) -> None:
    if not (len(imgs) == len(labels) == len(metadata_list)):
        print("\n[SYNC ERROR]", stage)
        print(" csv_path:", csv_path)
        print(" lens:", len(imgs), len(labels), len(metadata_list))
        raise AssertionError("len(imgs)!=len(labels)!=len(metadata_list)")


# ============================================================================
# 3) ?대?吏 由ъ궗?댁쫰 / ?꾩껜 ?뚯씠?꾨씪??# ============================================================================


def resize_img_height(img: np.ndarray, target_h: int = 304) -> np.ndarray:
    h, w, c = img.shape
    if h == target_h:
        return img
    zoom_h = target_h / h
    return zoom(img, (zoom_h, 1.0, 1.0), order=1).astype(np.float32)


def process_cropped_data(
    data_dir: str = "4. merge_data",
    output_dir: str = "1. hard_train_data",
    sort_by: str = "height",
    min_points: int = 200,
    max_points: int = 400,
    run_subdir: Optional[str] = None,
    roi_edit_dir: str = "5. edit_data",
) -> Optional[Path]:
    """edit_data/merge_data?먯꽌 CSV瑜??쎌뼱 bbox ?쇰꺼 ?ы븿 NPY ??? run_subdir=None?대㈃ YYYYMMDD_HHmm ?대뜑 ?앹꽦."""
    output_path = Path(output_dir)
    if output_path.is_absolute():
        base_output = output_path
    else:
        base_output = Path(current_dir) / output_path
    base_output.mkdir(parents=True, exist_ok=True)
    if run_subdir is None:
        run_subdir = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = base_output / run_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"저장 경로: {output_path}")

    K = 10
    LABEL_DIM = 1 + (3 * K * 4) + (3 * K)

    print("파단 데이터 파일 수집 중...")
    crop_files = collect_all_crop_files(data_dir, True)
    normal_files = collect_all_crop_files(data_dir, False)

    if normal_files:
        unique_files = len({f[0] for f in normal_files})
        print(f"  정상 데이터 파일: {unique_files}개")
        print(f"  정상 데이터 샘플: {len(normal_files)}개")

    if not crop_files and not normal_files:
        print("처리할 파일을 찾을 수 없습니다.")
        return None

    print(f"\n총 {len(crop_files)}개의 파단 파일, {len(normal_files)}개의 정상 샘플 발견")
    print("\n각 파일별 min/max로 정규화합니다...")

    imgs = []
    labels_cls = []
    labels = []
    metadata_list = []
    failed_files = []
    break_image_set = set()
    skip_prepare = 0
    skip_points = 0
    skip_no_roi = 0
    skip_no_bbox = 0

    # ---- ?뚮떒 泥섎━ ----
    if crop_files:
        for csv_path, project_name, poleid, label in tqdm(crop_files, desc="파단 데이터 처리"):
            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by=sort_by,
                feature_min_max=None,
            )
            if result is None:
                skip_prepare += 1
                continue
            img, metadata = result

            if metadata["original_length"] < min_points or metadata["original_length"] > max_points:
                skip_points += 1
                continue

            img = resize_img_height(img, target_h=img_height)
            roi_json_path = match_roi_json_from_csv(csv_path, edit_data_dir=roi_edit_dir)
            roi_info = load_roi_info_json(roi_json_path) if roi_json_path else None
            roi_bboxes = expand_rois_from_roi_info(roi_info)

            roi_dict = {0: [], 1: [], 2: []}
            for roi_idx, bbox in roi_bboxes:
                bbox_n = normalize_bbox_center_width(bbox, metadata["feature_min_max"])
                if bbox_n is None:
                    continue
                roi_dict[roi_idx].append(bbox_n)

            total = sum(len(v) for v in roi_dict.values())
            if total == 0:
                if roi_info is None:
                    skip_no_roi += 1
                else:
                    skip_no_bbox += 1
                continue

            bbox_tensor = np.zeros((3, K, 4), dtype=np.float32)
            mask_tensor = np.zeros((3, K), dtype=np.float32)
            for r in (0, 1, 2):
                bxs = roi_dict[r][:K]
                if len(bxs) > 0:
                    bbox_tensor[r, : len(bxs), :] = np.array(bxs, dtype=np.float32)
                    mask_tensor[r, : len(bxs)] = 1.0

            imgs.append(img)
            labels_cls.append(label)
            y_vec = np.zeros((LABEL_DIM,), dtype=np.float32)
            y_vec[0] = float(label)
            off_bbox = 1
            bbox_flat = bbox_tensor.reshape(-1).astype(np.float32)
            mask_flat = mask_tensor.reshape(-1).astype(np.float32)
            y_vec[off_bbox : off_bbox + bbox_flat.size] = bbox_flat
            y_vec[off_bbox + bbox_flat.size : off_bbox + bbox_flat.size + mask_flat.size] = mask_flat
            labels.append(y_vec)

            md = dict(metadata)
            md.update({
                "csv_path": str(csv_path),
                "project_name": project_name,
                "poleid": poleid,
                "img_height": img_height,
                "bbox_K": K,
                "bbox_total": int(total),
                "roi_bbox_counts": {r: len(roi_dict[r]) for r in (0, 1, 2)},
            })
            metadata_list.append(md)
            _check_sync("break_append", csv_path, imgs, labels, metadata_list)
            break_image_set.add(str(csv_path))

    break_sample_count = len(break_image_set)
    max_normal_samples = break_sample_count * 10
    print(f"\n파단 샘플 수: {break_sample_count}개")
    if crop_files and break_sample_count == 0:
        print("  [진단] 파단 샘플 0개 원인: prepare실패={}, 포인트범위초과={}, roi_info없음={}, bbox유효0={}".format(
            skip_prepare, skip_points, skip_no_roi, skip_no_bbox
        ))
        print("  roi_info 기준 경로: {}. 해당 경로에 *_OUT_processed_roi_info.json이 있는지 확인하세요.".format(roi_edit_dir))
    print(f"정상 샘플 최대 수: {max_normal_samples}개 (파단 샘플의 10배)")

    # ---- ?뺤긽 泥섎━ ----
    if normal_files:
        normal_kept = 0
        for csv_path, project_name, poleid, label in tqdm(normal_files, desc="정상 데이터 처리"):
            if normal_kept >= max_normal_samples:
                break
            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by=sort_by,
                feature_min_max=None,
            )
            if result is None:
                continue
            img, metadata = result
            if not (min_points <= metadata["original_length"] <= max_points):
                continue
            img = resize_img_height(img, target_h=img_height)

            bbox_tensor = np.zeros((3, K, 4), dtype=np.float32)
            mask_tensor = np.zeros((3, K), dtype=np.float32)
            y_vec = np.zeros((LABEL_DIM,), dtype=np.float32)
            y_vec[0] = 0.0
            off = 1
            bbox_flat = bbox_tensor.reshape(-1)
            mask_flat = mask_tensor.reshape(-1)
            y_vec[off : off + bbox_flat.size] = bbox_flat
            y_vec[off + bbox_flat.size : off + bbox_flat.size + mask_flat.size] = mask_flat

            imgs.append(img)
            labels_cls.append(0)
            labels.append(y_vec)
            md = dict(metadata)
            md.update({
                "csv_path": str(csv_path),
                "project_name": project_name,
                "poleid": poleid,
                "img_height": img_height,
                "bbox_K": K,
                "bbox_total": 0,
                "roi_bbox_counts": {0: 0, 1: 0, 2: 0},
            })
            metadata_list.append(md)
            _check_sync("normal_append", csv_path, imgs, labels, metadata_list)
            normal_kept += 1
        print(f"  최종 정상 샘플 수: {len([1 for c in labels_cls if c == 0])}개")

    if not imgs:
        print("생성된 시퀀스 데이터가 없습니다.")
        return None

    shapes = {}
    for im in imgs:
        shapes[im.shape] = shapes.get(im.shape, 0) + 1
    print("서로 다른 img.shape 개수:", len(shapes))
    for s, cnt in sorted(shapes.items(), key=lambda x: -x[1])[:10]:
        print(s, cnt)

    assert len(imgs) == len(labels) == len(metadata_list)
    X = np.array(imgs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    assert np.isfinite(y).all()

    print(f"\n시퀀스 데이터 생성 완료:")
    print(f"  총 샘플 수: {len(X)}")
    print(f"  시퀀스 형태: {X.shape}")
    print(f"  실패 파일 수: {len(failed_files)}개")
    print("\n[DEBUG] Break ROI/BBox 통계:")

    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.1, random_state=42, stratify=y[:, 0]
    )

    print(f"\n데이터 분할 (학습:테스트 = 9:1):")
    print(f"  학습 샘플 수: {len(X_train)}개")
    print(f"  테스트 샘플 수: {len(X_test)}개")

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "break_imgs_train.npy", X_train)
    np.save(train_dir / "break_labels_train.npy", y_train)
    np.save(test_dir / "break_imgs_test.npy", X_test)
    np.save(test_dir / "break_labels_test.npy", y_test)
    np.save(output_path / "break_imgs.npy", X)
    np.save(output_path / "break_labels.npy", y)

    metadata_file = output_path / "break_imgs_metadata.json"
    created_at = datetime.now().isoformat(timespec="seconds")
    metadata_dict = {
        "run_subdir": run_subdir,
        "created_at": created_at,
        "total_samples": len(metadata_list),
        "img_height": img_height,
        "feature_names": ["height", "degree", "x_value", "y_value", "z_value"],
        "normalization_method": "per_file",
        "sort_by": sort_by,
        "min_points": min_points,
        "samples": metadata_list,
        "failed_files": failed_files,
        "data_shape": list(X.shape),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

    print("\n데이터 저장 완료:")
    print(f"  학습 이미지: {train_dir / 'break_imgs_train.npy'}")
    print(f"  학습 라벨: {train_dir / 'break_labels_train.npy'}")
    print(f"  테스트 이미지: {test_dir / 'break_imgs_test.npy'}")
    print(f"  테스트 라벨: {test_dir / 'break_labels_test.npy'}")
    print(f"  메타데이터: {metadata_file}")

    break_samples = [i for i, l in enumerate(labels_cls) if l == 1]
    normal_samples = [i for i, l in enumerate(labels_cls) if l == 0]
    print("\n데이터 통계:")
    print(f"  총 샘플 수: {len(imgs)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  파단 샘플 비율: {len(break_samples)/len(imgs)*100:.2f}%")
    # 2李?Hard 紐⑤뜽(13. evaluate_hard_model_2nd.py)???꾪븳 ?꾩쿂由?罹먯떆???④퍡 ?앹꽦
    try:
        _generate_eval_cache_for_hard_model_2nd(
            output_path=output_path,
            data_dir=data_dir,
            min_points=min_points,
            max_points=max_points,
        )
    except Exception as e:
        print("경고: 2nd-stage eval 전처리 캐시 생성 중 오류:", e)

    return output_path


# ============================================================================
# 3-bis) 2李?Hard 紐⑤뜽 ?됯????꾩쿂由?罹먯떆 ?앹꽦
#   - 13. evaluate_hard_model_2nd.py?먯꽌 ?ъ슜?섎뒗 build_dataset? ?숈씪???뺥깭濡?#     11. evaluate_resnet_model/preprocessed_cache/ ?꾨옒??NPZ/PKL ???#   - ?대젃寃??대몢硫?13踰??ㅽ겕由쏀듃 ?ㅽ뻾 ??CSV ?ъ쟾泥섎━ ???罹먯떆瑜?諛붾줈 ?ъ슜 媛??#   - ?섏〈?깆쓣 以꾩씠湲??꾪빐, ?ш린?쒕뒗 ???뚯씪???뺤쓽???꾩쿂由??좏떥
#     (load_crop_csv, prepare_sequence_from_csv, collect_all_crop_files, resize_img_height)瑜?吏곸젒 ?ъ슜?쒕떎.
# ============================================================================


def _generate_eval_cache_for_hard_model_2nd(
    output_path: Path,
    data_dir: str = "4. merge_data",
    min_points: int = 200,
    max_points: int = 400,
) -> None:
    """
    2李?Hard 紐⑤뜽 ?됯? ?ㅽ겕由쏀듃(13. evaluate_hard_model_2nd.py)瑜??꾪븳
    ?꾩쿂由?罹먯떆(X, metas, csv_paths, labels)瑜?誘몃━ ?앹꽦.

    - 罹먯떆 寃쎈줈/?ㅻ뒗 13踰??ㅽ겕由쏀듃??湲곕낯媛믨낵 ?숈씪?섍쾶 留욎땄:
      11. evaluate_resnet_model/preprocessed_cache/<CACHE_KEY>.npz / .pkl
    - 13踰덉쓣 湲곕낯 ?듭뀡?쇰줈 ?ㅽ뻾?섎㈃ ??罹먯떆瑜?諛붾줈 ?ъ슜?섍퀬, ?꾩쿂由щ? ?앸왂?섍쾶 ??
    """
    print("\n[2nd-stage eval] 전처리 캐시 생성 시작...")
    imgs, metas, csv_paths, labels = [], [], [], []
    files: List[Tuple[str, str, str, int]] = []

    # break / normal 紐⑤몢 ?ы븿 (13踰덉쓽 湲곕낯 ?ㅼ젙怨??숈씪)
    files += collect_all_crop_files(data_dir, True)
    files += collect_all_crop_files(data_dir, False)

    if not files:
        print("[2nd-stage eval] 처리할 CSV 파일이 없어 캐시 생성을 건너뜁니다.")
        return

    for csv_path, _, _, label in tqdm(files, desc="[2nd-stage eval] 데이터 전처리"):
        r = prepare_sequence_from_csv(
            csv_path=csv_path,
            sort_by="height",
            feature_min_max=None,
        )
        if r is None:
            continue
        img, meta = r
        if not (min_points <= meta.get("original_length", 0) <= max_points):
            continue
        img = resize_img_height(img, target_h=img_height)
        imgs.append(img)
        metas.append(meta)
        csv_paths.append(csv_path)
        labels.append(int(label))

    if not imgs:
        print("[2nd-stage eval] 유효한 샘플이 없어 캐시 생성을 건너뜁니다.")
        return

    X = np.array(imgs, dtype=np.float32)

    # 캐시를 1. hard_train_data/<run_subdir>/eval_cache 아래에 저장
    cache_dir = output_path / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 13. evaluate_hard_model_2nd.py ??cache_key ?앹꽦 諛⑹떇怨??숈씪
    cache_key = f"{data_dir}_{min_points}_{max_points}_False_False_None".replace(" ", "_").replace(".", "_")
    cache_npz = cache_dir / f"{cache_key}.npz"
    cache_pkl = cache_dir / f"{cache_key}.pkl"

    np.savez_compressed(cache_npz, X=X)
    with open(cache_pkl, "wb") as f:
        pickle.dump((metas, csv_paths, labels), f)

    print("[2nd-stage eval] 전처리 캐시 생성 완료:", cache_npz)


# ============================================================================
# 4) ?ㅽ뻾
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="시퀀스 학습 데이터 준비 (bbox)")
    parser.add_argument(
        "--data-dir",
        default="4. merge_data",
        help="CSV 수집 디렉터리 (기본: 4. merge_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="1. hard_train_data",
        help="NPY 저장 기준 디렉터리 (기본: 1. hard_train_data)",
    )
    parser.add_argument(
        "--run-subdir",
        default=None,
        help="실행 하위 폴더명. None이면 현재 시각(YYYYMMDD_HHmm) 생성",
    )
    parser.add_argument(
        "--roi-edit-dir",
        default="5. edit_data",
        help="roi_info.json 수집 디렉터리 (기본: 5. edit_data)",
    )
    parser.add_argument("--sort-by", default="height", choices=["height", "degree"])
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=400)
    args = parser.parse_args()

    process_cropped_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sort_by=args.sort_by,
        min_points=args.min_points,
        max_points=args.max_points,
        run_subdir=args.run_subdir,
        roi_edit_dir=args.roi_edit_dir,
    )

