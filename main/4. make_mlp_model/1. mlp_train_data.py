#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Light/Hard 모델 출력을 결합해 MLP 학습 데이터셋을 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent


def configure_tf_runtime(device: str) -> None:
    """TensorFlow 실행 디바이스를 설정한다."""
    req = (device or "auto").lower()
    gpus = tf.config.list_physical_devices("GPU")

    if req == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[정보][TF] CPU 모드로 실행합니다. (GPU 비활성화)")
        return

    if req == "gpu":
        if not gpus:
            raise RuntimeError("device=gpu가 지정됐지만 사용 가능한 GPU가 없습니다.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[정보][TF] GPU 모드로 실행합니다. (GPU 수: {len(gpus)})")
        return

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[정보][TF] auto 모드: GPU 사용 (GPU 수: {len(gpus)})")
    else:
        print("[정보][TF] auto 모드: CPU 사용")


def get_latest_hard_data_run(base: Path) -> Path:
    """가장 최신 Hard 데이터 실행 디렉터리를 찾는다."""
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


def _resolve_best_selection_dir(best_model_dir: Path, required_first_stage_run: str | None = None) -> Path:
    """베스트 모델 별칭 디렉터리에서 실제 selection 파일 위치를 찾는다."""
    if required_first_stage_run:
        by_first_stage_dir = best_model_dir / "by_first_stage" / required_first_stage_run
        if (by_first_stage_dir / "best_model_selection.json").exists():
            return by_first_stage_dir

    overall_best_dir = best_model_dir / "overall_best"
    if (overall_best_dir / "best_model_selection.json").exists():
        return overall_best_dir

    direct_selection = best_model_dir / "best_model_selection.json"
    if direct_selection.exists():
        return best_model_dir

    raise FileNotFoundError(f"Best model selection file not found under: {best_model_dir}")


def get_best_hard_model_info(best_model_dir: Path, required_first_stage_run: str | None = None) -> Dict:
    """best_hard_model 디렉터리에서 선택된 모델 정보를 읽는다."""
    selection_base_dir = _resolve_best_selection_dir(best_model_dir, required_first_stage_run=required_first_stage_run)
    selection_file = selection_base_dir / "best_model_selection.json"
    
    with open(selection_file, 'r', encoding='utf-8') as f:
        selection_info = json.load(f)
    
    selected_run = selection_info["selected"]["model_run"]
    run_dir = selection_base_dir / selected_run
    
    if not run_dir.exists():
        print(f"[경고] 정확히 일치하는 run 디렉터리를 찾지 못했습니다: {selected_run}")
        print("[정보] 유사한 디렉터리를 탐색합니다.")
        
        candidates = []
        for d in selection_base_dir.iterdir():
            if d.is_dir() and selected_run.startswith(d.name):
                candidates.append(d)
        
        if candidates:
            run_dir = max(candidates, key=lambda x: len(x.name))
            actual_run_name = run_dir.name
            print(f"[정보] 유사 디렉터리를 사용합니다: {actual_run_name}")
        else:
            raise FileNotFoundError(f"No similar model run directory found for: {selected_run}")
    else:
        actual_run_name = selected_run
    
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    return {
        "selection_info": selection_info,
        "selection_base_dir": selection_base_dir,
        "selection_file": selection_file,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "selected_run": selected_run,
        "actual_run_name": actual_run_name
    }


def flatten_binary_output(pred: np.ndarray) -> np.ndarray:
    """이진 분류 출력을 1차원 배열로 평탄화한다."""
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.astype(np.float32)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0].astype(np.float32)
    return pred.reshape(pred.shape[0], -1)[:, 0].astype(np.float32)


def flatten_axis_conf(pred: np.ndarray) -> np.ndarray:
    """축별 확률 출력을 1차원 배열로 평탄화한다(최댓값 사용)."""
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.astype(np.float32)
    if pred.ndim == 2:
        return np.max(pred, axis=1).astype(np.float32)
    return pred.reshape(pred.shape[0], -1).max(axis=1).astype(np.float32)


def _resolve_axis_model_path(model_dir: Path, axis: str, candidates: List[str]) -> Path:
    for filename in candidates:
        model_path = model_dir / filename.format(axis=axis)
        if model_path.exists():
            return model_path
    tried = ", ".join(filename.format(axis=axis) for filename in candidates)
    raise FileNotFoundError(f"Axis model file not found in {model_dir}: {tried}")


def collect_test_csv_paths(run_dir: Path, n_test: int) -> List[str]:
    """테스트 데이터 CSV 경로를 메타데이터에서 수집한다."""
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


def create_mlp_dataset(
    hard_data_run: Path,
    light_model_path: Path,
    hard1_model_dir: Path,
    hard2_model_dir: Path,
    weight_light: float = 0.4,
    weight_hard1: float = 0.3,
    weight_hard2: float = 0.3,
    weight_x: float = 0.34,
    weight_y: float = 0.33,
    weight_z: float = 0.33,
    val_size: float = 0.3,
    seed: int = 42
) -> Dict:
    """
    MLP 학습용 데이터셋을 생성한다.

    Args:
        hard_data_run: Hard 데이터 실행 디렉터리
        light_model_path: Light 모델 경로
        hard1_model_dir: Hard 1차 모델 디렉터리
        hard2_model_dir: Hard 2차 모델 디렉터리
        weight_light: Light 가중치
        weight_hard1: Hard1 가중치
        weight_hard2: Hard2 가중치
        weight_x: X축 가중치
        weight_y: Y축 가중치
        weight_z: Z축 가중치
        val_size: 검증 데이터 비율
        seed: 난수 시드

    Returns:
        Dict: 생성된 데이터셋과 메타데이터
    """
    x_test_path = hard_data_run / "test" / "break_imgs_test.npy"
    y_test_path = hard_data_run / "test" / "break_labels_test.npy"
    
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(f"Missing hard test data: {x_test_path}, {y_test_path}")

    X_test = np.load(x_test_path).astype(np.float32)
    y_test_raw = np.load(y_test_path)
    y_true = y_test_raw[:, 0].astype(np.int32) if y_test_raw.ndim == 2 else y_test_raw.astype(np.int32)
    
    csv_paths = collect_test_csv_paths(hard_data_run, len(X_test))
    sample_ids = [Path(p).name.replace("_OUT_processed.csv", "") if p else f"idx_{i}" for i, p in enumerate(csv_paths)]

    hard1_conf_x_path = _resolve_axis_model_path(hard1_model_dir, "x", ["best_{axis}.keras", "conf_{axis}.keras"])
    hard1_conf_y_path = _resolve_axis_model_path(hard1_model_dir, "y", ["best_{axis}.keras", "conf_{axis}.keras"])
    hard1_conf_z_path = _resolve_axis_model_path(hard1_model_dir, "z", ["best_{axis}.keras", "conf_{axis}.keras"])

    hard2_conf_x_path = _resolve_axis_model_path(hard2_model_dir, "x", ["conf_{axis}.keras", "best_{axis}.keras"])
    hard2_conf_y_path = _resolve_axis_model_path(hard2_model_dir, "y", ["conf_{axis}.keras", "best_{axis}.keras"])
    hard2_conf_z_path = _resolve_axis_model_path(hard2_model_dir, "z", ["conf_{axis}.keras", "best_{axis}.keras"])
    
    model_paths = [
        light_model_path,
        hard1_conf_x_path, hard1_conf_y_path, hard1_conf_z_path,
        hard2_conf_x_path, hard2_conf_y_path, hard2_conf_z_path
    ]
    
    for p in model_paths:
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

    print("[정보] 모델을 로드합니다...")
    light_model = tf.keras.models.load_model(light_model_path, compile=False)
    
    hard1_conf_x_model = tf.keras.models.load_model(hard1_conf_x_path, compile=False)
    hard1_conf_y_model = tf.keras.models.load_model(hard1_conf_y_path, compile=False)
    hard1_conf_z_model = tf.keras.models.load_model(hard1_conf_z_path, compile=False)
    
    hard2_conf_x_model = tf.keras.models.load_model(hard2_conf_x_path, compile=False)
    hard2_conf_y_model = tf.keras.models.load_model(hard2_conf_y_path, compile=False)
    hard2_conf_z_model = tf.keras.models.load_model(hard2_conf_z_path, compile=False)

    print("[정보] hard test 데이터 예측을 수행합니다...")
    light_prob = flatten_binary_output(light_model.predict(X_test, verbose=0))
    
    hard1_conf_x = flatten_axis_conf(hard1_conf_x_model.predict(X_test, verbose=0))
    hard1_conf_y = flatten_axis_conf(hard1_conf_y_model.predict(X_test, verbose=0))
    hard1_conf_z = flatten_axis_conf(hard1_conf_z_model.predict(X_test, verbose=0))
    
    hard2_conf_x = flatten_axis_conf(hard2_conf_x_model.predict(X_test, verbose=0))
    hard2_conf_y = flatten_axis_conf(hard2_conf_y_model.predict(X_test, verbose=0))
    hard2_conf_z = flatten_axis_conf(hard2_conf_z_model.predict(X_test, verbose=0))

    axis_weight_sum = weight_x + weight_y + weight_z
    total_weight = weight_light + weight_hard1 + weight_hard2
    
    if axis_weight_sum <= 0:
        raise ValueError("Axis weights sum must be > 0")
    
    wx = weight_x / axis_weight_sum
    wy = weight_y / axis_weight_sum
    wz = weight_z / axis_weight_sum

    hard1_score = (wx * hard1_conf_x) + (wy * hard1_conf_y) + (wz * hard1_conf_z)
    
    hard2_score = (wx * hard2_conf_x) + (wy * hard2_conf_y) + (wz * hard2_conf_z)
    
    w_light = weight_light / total_weight
    w_hard1 = weight_hard1 / total_weight
    w_hard2 = weight_hard2 / total_weight
    
    fused_score = (w_light * light_prob) + (w_hard1 * hard1_score) + (w_hard2 * hard2_score)
    
    light_hard1_gap = np.abs(light_prob - hard1_score)
    light_hard2_gap = np.abs(light_prob - hard2_score)
    hard1_hard2_gap = np.abs(hard1_score - hard2_score)

    features = np.column_stack([
        light_prob,
        hard1_conf_x, hard1_conf_y, hard1_conf_z,
        hard2_conf_x, hard2_conf_y, hard2_conf_z,
        hard1_score, hard2_score, fused_score,
        light_hard1_gap, light_hard2_gap, hard1_hard2_gap,
    ]).astype(np.float32)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        features,
        y_true,
        np.arange(len(y_true)),
        test_size=val_size,
        random_state=seed,
        stratify=y_true,
    )

    feature_names = [
        'light_prob', 
        'hard1_conf_x', 'hard1_conf_y', 'hard1_conf_z',
        'hard2_conf_x', 'hard2_conf_y', 'hard2_conf_z',
        'hard1_score', 'hard2_score', 'fused_score',
        'light_hard1_gap', 'light_hard2_gap', 'hard1_hard2_gap'
    ]
    
    dataset_df = pd.DataFrame(features, columns=feature_names)
    dataset_df['y_true'] = y_true
    dataset_df['sample_id'] = sample_ids
    dataset_df['csv_path'] = csv_paths

    print("[정보] 데이터셋 생성 완료")
    print(f"  총 샘플 수: {len(features)}")
    print(f"  피처 수: {features.shape[1]}")
    print(f"  학습 샘플 수: {len(X_train)}")
    print(f"  검증 샘플 수: {len(X_val)}")
    print(f"  정상 샘플 수: {(y_true == 0).sum()}")
    print(f"  파단 샘플 수: {(y_true == 1).sum()}")

    return {
        'features': features,
        'labels': y_true,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'dataset_df': dataset_df,
        'feature_names': feature_names,
        'sample_ids': sample_ids,
        'csv_paths': csv_paths,
        'metadata': {
            'hard_data_run': str(hard_data_run),
            'light_model_path': str(light_model_path),
            'hard1_model_dir': str(hard1_model_dir),
            'hard2_model_dir': str(hard2_model_dir),
            'weights': {
                'light': weight_light,
                'hard1': weight_hard1,
                'hard2': weight_hard2,
                'normalized_light': w_light,
                'normalized_hard1': w_hard1,
                'normalized_hard2': w_hard2,
                'axis_x': wx,
                'axis_y': wy,
                'axis_z': wz,
            },
            'val_size': val_size,
            'seed': seed,
            'total_samples': len(features),
            'feature_count': features.shape[1],
            'class_distribution': {
                'normal': int((y_true == 0).sum()),
                'break': int((y_true == 1).sum()),
            }
        }
    }


def save_dataset(dataset_dict: Dict, output_dir: Path, run_name: str = None) -> Path:
    """생성한 데이터셋을 파일로 저장한다."""
    if run_name is None:
        import datetime as dt
        run_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "features.npy", dataset_dict['features'])
    np.save(save_dir / "labels.npy", dataset_dict['labels'])
    np.save(save_dir / "X_train.npy", dataset_dict['X_train'])
    np.save(save_dir / "X_val.npy", dataset_dict['X_val'])
    np.save(save_dir / "y_train.npy", dataset_dict['y_train'])
    np.save(save_dir / "y_val.npy", dataset_dict['y_val'])
    np.save(save_dir / "idx_train.npy", dataset_dict['idx_train'])
    np.save(save_dir / "idx_val.npy", dataset_dict['idx_val'])
    
    dataset_dict['dataset_df'].to_csv(save_dir / "mlp_dataset.csv", index=False, encoding="utf-8-sig")

    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_dict['metadata'], f, indent=2, ensure_ascii=False)

    with (save_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_dict['feature_names'], f, indent=2, ensure_ascii=False)

    print(f"[정보] MLP 학습 데이터셋 저장 완료: {save_dir}")
    print("[정보] 저장 파일:")
    print(f"  - features.npy ({dataset_dict['features'].shape})")
    print(f"  - X_train.npy ({dataset_dict['X_train'].shape})")
    print(f"  - X_val.npy ({dataset_dict['X_val'].shape})")
    print("  - mlp_dataset.csv")
    print("  - metadata.json")
    print("  - feature_names.json")
    
    return save_dir


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="MLP 데이터셋 생성 - best_hard_model_1st, best_hard_model_2nd 사용")
    parser.add_argument("--hard-data-run", default=None, help="hard data run name (default: latest)")
    parser.add_argument("--light-model-dir", default="best_light_model", help="light model directory")
    parser.add_argument("--hard1-model-dir", default="best_hard_model_1st", help="hard1 best model directory")
    parser.add_argument("--hard2-model-dir", default="best_hard_model_2nd", help="hard2 best model directory")
    parser.add_argument("--output-dir", default="1. mlp_train_data", help="output directory")
    parser.add_argument("--run-name", default=None, help="run name for output")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="TF device")
    
    args = parser.parse_args()
    
    configure_tf_runtime(args.device)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    hard_data_base = CURRENT_DIR.parent / "3. make_hard_model" / "1. hard_train_data"
    if args.hard_data_run:
        hard_data_run = hard_data_base / args.hard_data_run
    else:
        hard_data_run = get_latest_hard_data_run(hard_data_base)
    
    light_base_dir = CURRENT_DIR.parent / "2. make_light_model" / args.light_model_dir
    light_info = get_best_hard_model_info(light_base_dir)
    light_model_path = light_info['checkpoints_dir'] / "best.keras"
    
    hard1_base_dir = CURRENT_DIR.parent / "3. make_hard_model" / args.hard1_model_dir
    hard2_base_dir = CURRENT_DIR.parent / "3. make_hard_model" / args.hard2_model_dir
    
    print("=== Best Models Information ===")
    
    print(f"Light Selected Run: {light_info['selected_run']}")
    if light_info['selected_run'] != light_info['actual_run_name']:
        print(f"Light Actual Directory: {light_info['actual_run_name']}")
    print(f"Light Metrics: {light_info['selection_info']['selected']['metrics']}")
    
    hard1_info = get_best_hard_model_info(hard1_base_dir)
    print(f"Hard1 Selected Run: {hard1_info['selected_run']}")
    if hard1_info['selected_run'] != hard1_info['actual_run_name']:
        print(f"Hard1 Actual Directory: {hard1_info['actual_run_name']}")
    print(f"Hard1 Metrics: {hard1_info['selection_info']['selected']['metrics']}")
    hard1_model_dir = hard1_info['checkpoints_dir']
    
    hard2_info = get_best_hard_model_info(
        hard2_base_dir,
        required_first_stage_run=hard1_info['actual_run_name'],
    )
    print(f"Hard2 Selected Run: {hard2_info['selected_run']}")
    if hard2_info['selected_run'] != hard2_info['actual_run_name']:
        print(f"Hard2 Actual Directory: {hard2_info['actual_run_name']}")
    print(f"Hard2 Selection Base: {hard2_info['selection_base_dir']}")
    print(f"Hard2 Metrics: {hard2_info['selection_info']['selected']['metrics']}")
    hard2_model_dir = hard2_info['checkpoints_dir']
    
    print("===================================\n")
    
    dataset_dict = create_mlp_dataset(
        hard_data_run=hard_data_run,
        light_model_path=light_model_path,
        hard1_model_dir=hard1_model_dir,
        hard2_model_dir=hard2_model_dir,
        weight_light=0.4,
        weight_hard1=0.3,
        weight_hard2=0.3,
        weight_x=0.34,
        weight_y=0.33,
        weight_z=0.33,
        val_size=0.3,
        seed=args.seed
    )
    
    dataset_dict['metadata']['light_model_info'] = light_info['selection_info']
    dataset_dict['metadata']['hard1_model_info'] = hard1_info['selection_info']
    dataset_dict['metadata']['hard2_model_info'] = hard2_info['selection_info']
    dataset_dict['metadata']['hard2_selection_base_dir'] = str(hard2_info['selection_base_dir'])
    
    output_dir = CURRENT_DIR / args.output_dir
    save_dataset(dataset_dict, output_dir, args.run_name)
    print(f"\n[정보] 데이터셋 저장 경로: {output_dir}")
    
    print("\n=== 데이터셋 요약 ===")
    print(f"Feature names: {dataset_dict['feature_names']}")
    print(f"Dataset shape: {dataset_dict['features'].shape}")
    print(f"Train/Val split: {len(dataset_dict['X_train'])}/{len(dataset_dict['X_val'])}")
    print(f"Light Model: {light_info['actual_run_name']}")
    print(f"Hard1 Model: {hard1_info['actual_run_name']}")
    print(f"Hard2 Model: {hard2_info['actual_run_name']}")
    print("========================")


if __name__ == "__main__":
    main()
