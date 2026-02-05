"""시퀀스 학습 데이터 준비 (CSV → NPY) - Light 모델용 학습/테스트 데이터 생성"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# ✅ 현재 스크립트 디렉토리를 기준으로 경로 설정
# 스크립트 파일의 위치를 기준으로 하므로 실행 위치와 무관하게 동작합니다
current_dir = os.path.dirname(os.path.abspath(__file__))

img_height = 304


# ============================================================================
# 1) CSV 로드/이미지 생성 유틸
# ============================================================================

def load_crop_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """크롭된 CSV 파일 로드."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def prepare_sequence_from_csv(
    csv_path: str,
    sort_by: str = 'height',
    feature_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    max_height: Optional[int] = None
) -> Optional[Tuple[np.ndarray, Dict]]:
    """CSV 파일에서 이미지 데이터 생성 (height, degree, x, y, z 포함, 각각 0~1 정규화)."""
    df = load_crop_csv(csv_path)
    if df is None:
        return None
    
    required_cols = ['height', 'degree', 'x_value', 'y_value', 'z_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None

    # 정렬 : 전주 높이 정렬이 기본
    if sort_by == 'height':
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)
    elif sort_by == 'degree':
        df = df.sort_values(['degree', 'height']).reset_index(drop=True)
    else:
        df = df.sort_values(['height', 'degree']).reset_index(drop=True)

    # 파일별 min/max
    if feature_min_max is None:
        feature_min_max = {
            'height': (df['height'].min(), df['height'].max()),
            'degree': (df['degree'].min(), df['degree'].max()),
            'x_value': (df['x_value'].min(), df['x_value'].max()),
            'y_value': (df['y_value'].min(), df['y_value'].max()),
            'z_value': (df['z_value'].min(), df['z_value'].max()),
        }

    # 0~1 정규화
    value_cols = ['x_value', 'y_value', 'z_value']
    for col in value_cols:
        vmin, vmax = feature_min_max[col]
        if vmax > vmin:
            df[col] = (df[col].astype(np.float32) - vmin) / (vmax - vmin)
        else:
            df[col] = 0.0

    # height/degree "값 그대로"를 축으로 grid 생성 (pivot 사용)
    heights = np.sort(df['height'].unique())
    degrees = np.arange(90.0, 180.0 + 5.0, 5.0, dtype=np.float32)  # 19개 고정
    H, W = len(heights), len(degrees)

    df['degree'] = (np.round(df['degree'] / 5.0) * 5.0).astype(np.float32)

    dmin, dmax = df['degree'].min(), df['degree'].max()

    if dmax <= 90.0:            # 0~90  -> +90
        df['degree'] += 90.0
    elif dmin >= 180.0 and dmax <= 270.0:  # 180~270 -> -90
        df['degree'] -= 90.0
    elif dmin >= 270.0:          # 270~360 -> -180
        df['degree'] -= 180.0

    def make_grid(col: str) -> np.ndarray:
        g = (df.pivot_table(index='height', columns='degree', values=col, aggfunc='mean')
               .reindex(index=heights, columns=degrees)
               .to_numpy(dtype=np.float32))
        return np.nan_to_num(g, nan=0.0)

    x_grid = make_grid('x_value')
    y_grid = make_grid('y_value')
    z_grid = make_grid('z_value')

    img = np.stack([x_grid, y_grid, z_grid], axis=-1).astype(np.float32)

    metadata = {
        'grid_shape': img.shape,            # (H, W, 3)
        'num_points': int(len(df)),
        'original_length': int(len(df)),
        'unique_heights': int(H),
        'unique_degrees': int(W),
        'height_values': heights.tolist(),  # "값 그대로" 축
        'degree_values': degrees.tolist(),  # "값 그대로" 축
        'feature_min_max': {k: list(v) for k, v in feature_min_max.items()},
    }

    return img, metadata


def resize_img_height(img: np.ndarray, target_h: int = 304) -> np.ndarray:
    """이미지 높이를 target_h로 리사이즈."""
    h, w, c = img.shape
    if h == target_h:
        return img
    zoom_h = target_h / h
    return zoom(img, (zoom_h, 1.0, 1.0), order=1).astype(np.float32)  # W/C는 그대로


# ============================================================================
# 2) 파일 수집
# ============================================================================

def collect_break_files_from_edit_data(
    edit_data_dir: str = "5. edit_data",
    merge_data_dir: str = "4. merge_data",
) -> List[Tuple[str, str, str, int]]:
    """5. edit_data/break에서 roi_info를 기준으로 수집. deleted=True 인 것은 제외.
    CSV는 5. edit_data 동일 경로 우선, 없으면 4. merge_data/break 동일 project/pole 에서 찾음."""
    edit_break = Path(current_dir) / edit_data_dir / "break"
    merge_break = Path(current_dir) / merge_data_dir / "break"
    crop_files = []

    if not edit_break.exists():
        print(f"경고: 파단(edit) 디렉토리를 찾을 수 없습니다: {edit_break}")
        print(f"  current_dir={current_dir!r}, edit_data_dir={edit_data_dir!r}")
        return crop_files

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
                    with open(roi_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    if info.get("deleted") is True:
                        continue
                except Exception:
                    continue
                # roi stem 예: "3242H961_1_OUT_processed_roi_info" → CSV "3242H961_1_OUT_processed.csv"
                # 끝의 "_roi_info"만 제거 (_processed_roi_info 제거하면 "3242H961_1_OUT.csv"가 됨)
                stem = roi_path.stem
                if stem.endswith("_roi_info"):
                    csv_base = stem[:-len("_roi_info")] + ".csv"
                else:
                    csv_base = stem.replace("_roi_info", "") + ".csv"
                csv_in_edit = pole_dir / csv_base
                csv_in_merge = merge_break / project_name / poleid / csv_base
                csv_path = None
                if csv_in_edit.exists():
                    csv_path = csv_in_edit
                elif merge_break.exists() and csv_in_merge.exists():
                    csv_path = csv_in_merge
                if csv_path is not None:
                    crop_files.append((str(csv_path), project_name, poleid, 1))

    return crop_files


def collect_all_crop_files(data_dir: str, is_break: bool) -> List[Tuple[str, str, str, int]]:
    """4. merge_data/break 또는 normal에서 CSV 파일 수집. (파단은 collect_break_files_from_edit_data 사용 권장)"""
    base_dir = Path(current_dir) / data_dir
    base_dir = base_dir / ("break" if is_break else "normal")

    crop_files = []

    if not base_dir.exists():
        print(f"경고: 데이터 디렉토리를 찾을 수 없습니다: {base_dir}")
        return crop_files

    for project_dir in base_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        for pole_dir in project_dir.iterdir():
            if not pole_dir.is_dir():
                continue
            poleid = pole_dir.name
            patterns = ["*_OUT_processed.csv"] if is_break else ["*_OUT_processed.csv", "*_normal_processed.csv"]

            for pat in patterns:
                for csv_file in pole_dir.glob(pat):
                    crop_files.append((str(csv_file), project_name, poleid, 1 if is_break else 0))

    return crop_files


# ============================================================================
# 3) 전체 처리 파이프라인 (CSV → NPY 저장)
# ============================================================================

def process_cropped_data(
    data_dir: str = "4. merge_data",
    output_dir: str = "6. light_train_data",
    break_data_dir: str = "5. edit_data",
    sort_by: str = 'height',
    min_points: int = 200,
    max_points: int = 400,
    run_subdir: Optional[str] = None,
):
    """데이터를 처리하여 6. light_train_data/<날짜시간>에 이미지·라벨 NPY 저장.
    파단: break_data_dir(5. edit_data)의 roi_info 기준, deleted 아닌 것만. CSV는 동일 경로 또는 data_dir(4. merge_data)에서 조회.
    정상: data_dir(4. merge_data)/normal에서 수집.
    run_subdir: None이면 현재 시각 기준 YYYYMMDD_HHmm 폴더 사용.
    """
    base_output = Path(current_dir) / output_dir
    base_output.mkdir(parents=True, exist_ok=True)
    if run_subdir is None:
        run_subdir = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = base_output / run_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"저장 경로: {output_path}")

    print("파단 데이터 파일 수집 중 (5. edit_data, 삭제처리 제외)...")
    crop_files = collect_break_files_from_edit_data(
        edit_data_dir=break_data_dir,
        merge_data_dir=data_dir,
    )
    print("정상 데이터 파일 수집 중 (4. merge_data)...")
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
    metadata_list = []
    failed_files = []

    # ---- 파단 처리 ----
    if crop_files:
        for csv_path, project_name, poleid, label in tqdm(crop_files, desc="파단 데이터 처리"):
            csv_file_path = Path(csv_path)

            result = prepare_sequence_from_csv(
                csv_path=csv_path,
                sort_by=sort_by,
                feature_min_max=None
            )

            if result is None:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '데이터 로드 실패 또는 형식 오류'
                })
                continue

            img, metadata = result

            if metadata['original_length'] < min_points or metadata['original_length'] > max_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 or 과다 ({metadata['original_length']})"
                })
                continue

            img = resize_img_height(img, target_h=304)

            imgs.append(img)
            labels_cls.append(label)

            metadata.update({
                'csv_path': csv_path,
                'project_name': project_name,
                'poleid': poleid,
                'img_height': img_height
            })
            metadata_list.append(metadata)

    break_sample_count = sum(1 for l in labels_cls if l == 1)
    max_normal_samples = break_sample_count * 10
    print(f"\n파단 샘플 수: {break_sample_count}개")
    print(f"정상 샘플 최대 수: {max_normal_samples}개 (파단 샘플의 10배)")

    # ---- 정상 처리 ----
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
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': '데이터 로드 실패 또는 형식 오류'
                })
                continue

            img, metadata = result

            if metadata['original_length'] < min_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"데이터 포인트 부족 ({metadata['original_length']} < {min_points})"
                })
                continue

            if metadata['original_length'] >= max_points:
                failed_files.append({
                    'csv_path': csv_path,
                    'project_name': project_name,
                    'poleid': poleid,
                    'reason': f"정상 데이터 포인트 수가 {max_points}개 이상 ({metadata['original_length']} >= {max_points})"
                })
                continue

            img = resize_img_height(img, target_h=304)

            imgs.append(img)
            labels_cls.append(label)

            normal_kept += 1

            metadata.update({
                'csv_path': csv_path,
                'project_name': project_name,
                'poleid': poleid,
            })
            metadata_list.append(metadata)

        # 정상 샘플이 초과면 샘플링
        normal_indices = [i for i, l in enumerate(labels_cls) if l == 0]
        if len(normal_indices) > max_normal_samples:
            print(f"  생성된 정상 샘플: {len(normal_indices)}개 → {max_normal_samples}개로 샘플링")
            indices_to_keep = np.random.choice(normal_indices, size=max_normal_samples, replace=False)
            break_indices = [i for i, l in enumerate(labels_cls) if l == 1]
            all_indices = break_indices + list(indices_to_keep)

            imgs = [imgs[i] for i in all_indices]
            labels_cls = [labels_cls[i] for i in all_indices]
            metadata_list = [metadata_list[i] for i in all_indices]

        print(f"  최종 정상 샘플 수: {len([l for l in labels_cls if l == 0])}개")

    if not imgs:
        print("생성된 시퀀스 데이터가 없습니다.")
        return None

    X = np.array(imgs)
    # 7. make_light_model.py 가 y[:, 0]을 쓰므로 (N, 1) 형태로 저장
    y = np.array(labels_cls, dtype=np.int64).reshape(-1, 1)

    print(f"\n시퀀스 데이터 생성 완료: 총 샘플 수={len(X)}, 형태={X.shape}, 실패 파일={len(failed_files)}개")

    # 9:1 분할 (클래스가 하나일 때는 stratify 비활성)
    indices = np.arange(len(X))
    y_flat = y.ravel()
    stratify_arg = y_flat if len(np.unique(y_flat)) > 1 else None
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.1, random_state=42, stratify=stratify_arg
    )

    print(f"\n데이터 분할 (학습:테스트 = 9:1):")
    print(f"  학습 샘플 수: {len(X_train)}개")
    print(f"  테스트 샘플 수: {len(X_test)}개")

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / "break_imgs_train.npy", X_train)
    np.save(train_dir / "break_labels_train.npy", y_train)   # (N, 1) 유지

    np.save(test_dir / "break_imgs_test.npy", X_test)
    np.save(test_dir / "break_labels_test.npy", y_test)   # (N, 1) 유지

    np.save(output_path / "break_imgs.npy", X)
    np.save(output_path / "break_labels.npy", y)   # (N, 1)

    metadata_file = output_path / "break_imgs_metadata.json"
    created_at = datetime.now().isoformat(timespec='seconds')
    metadata_dict = {
        'run_subdir': run_subdir,
        'created_at': created_at,
        'total_samples': len(metadata_list),
        'img_height': img_height,
        'feature_names': ['height', 'degree', 'x_value', 'y_value', 'z_value'],
        'normalization_method': 'per_file',
        'sort_by': sort_by,
        'min_points': min_points,
        'samples': metadata_list,
        'failed_files': failed_files,
        'data_shape': list(X.shape),
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

    print("\n데이터 저장 완료:")
    print(f"  학습 시퀀스: {train_dir / 'break_imgs_train.npy'}")
    print(f"  학습 라벨: {train_dir / 'break_labels_train.npy'}")
    print(f"  테스트 시퀀스: {test_dir / 'break_imgs_test.npy'}")
    print(f"  테스트 라벨: {test_dir / 'break_labels_test.npy'}")
    print(f"  메타데이터: {metadata_file}")

    break_samples = [i for i, l in enumerate(labels_cls) if l == 1]
    normal_samples = [i for i, l in enumerate(labels_cls) if l == 0]
    print("\n데이터 통계:")
    print(f"  총 샘플 수: {len(imgs)}개")
    print(f"    - 파단 샘플: {len(break_samples)}개 (라벨 1)")
    print(f"    - 정상 샘플: {len(normal_samples)}개 (라벨 0)")
    print(f"  파단 샘플 비율: {len(break_samples)/len(imgs)*100:.2f}%")
    return output_path


def get_latest_run_dir(base_dir: str = "6. light_train_data") -> Optional[Path]:
    """6. light_train_data 안에서 YYYYMMDD_HHmm 형식의 가장 최근 실행 폴더 경로 반환."""
    base = Path(current_dir) / base_dir
    if not base.exists():
        return None
    subdirs = [d for d in base.iterdir() if d.is_dir() and len(d.name) == 12 and d.name[:8].isdigit() and d.name[9:].isdigit() and d.name[8] == '_']
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.name)


# ============================================================================
# 4) 실행
# ============================================================================

if __name__ == "__main__":
    print("current_dir:", current_dir)
    
    # 필요하면 아래 파라미터만 조정해서 실행하세요.
    # run_subdir=None 이면 현재 시각(YYYYMMDD_HHmm)으로 폴더 생성
    saved_run = process_cropped_data(
        data_dir="4. merge_data",
        output_dir="6. light_train_data",
        break_data_dir="5. edit_data",
        sort_by="height",
        min_points=200,
        max_points=400,
        run_subdir=None,
    )
    
    # 저장된 NPY 빠른 확인(배치) — 방금 저장한 run 또는 최신 run 사용
    print("\n" + "="*50)
    print("저장된 NPY 빠른 확인")
    print("="*50)
    run_dir = saved_run if saved_run else get_latest_run_dir()
    if run_dir:
        train_seq = run_dir / "train" / "break_imgs_train.npy"
        train_lab = run_dir / "train" / "break_labels_train.npy"
    else:
        train_seq = train_lab = None
    
    if train_seq and train_seq.exists() and train_lab and train_lab.exists():
        print(f"로드 경로: {run_dir}")
        X = np.load(train_seq)
        y = np.load(train_lab)
        bs = 32
        Xb, yb = X[:bs], y[:bs]
        print("X batch shape:", Xb.shape, Xb.dtype)
        print("y batch shape:", yb.shape, yb.dtype)
        print("y batch counts:", np.unique(yb, return_counts=True))
        print("X[0] first 5 rows:\n", Xb[0, :5, :])
    else:
        print("학습 데이터 파일을 찾을 수 없습니다. 먼저 process_cropped_data()를 실행하세요.")
