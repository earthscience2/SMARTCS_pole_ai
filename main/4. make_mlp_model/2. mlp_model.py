# -*- coding: utf-8 -*-
"""MLP Final Decision Model Training with Best Model Management."""

import os
import sys
import subprocess
import shutil
import datetime
import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Windows 인코딩 문제 해결
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

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
_run_local = "--local" in sys.argv or sys.platform != "win32"
if _run_local:
    if "--local" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--local"]
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parent
    _sh_path = Path(_script_dir) / "2. mlp_model_gpu.sh"
    if _sh_path.exists():
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = (
            "/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")
        ) if _drive else str(_abs).replace("\\", "/")
        print("[정보] WSL2 GPU 학습 스크립트를 실행합니다:", _wsl_path)
        arg_str = " ".join(shlex.quote(a) for a in sys.argv[1:])
        cmd = f"bash {shlex.quote(_wsl_path)} {arg_str}".strip()
        ret = subprocess.run(["wsl", "bash", "-lc", cmd], cwd=str(_project_root))
        sys.exit(ret.returncode)

# ============================================================================
# ============================================================================
USER_OPTIONS = {
    "hidden_layers": [64, 32, 16],  # MLP 은닉층 구조
    "alpha": 1e-4,                  # L2 정규화 계수
    "max_iter": 1200,               # 최대 반복 횟수
    "early_stopping": True,         # 조기 종료 사용
    "validation_fraction": 0.2,     # 조기종료용 내부 검증 비율
    "n_iter_no_change": 25,         # 조기 종료 patience
    "val_size": 0.3,                # 홀드아웃 검증 비율
    "seed": 42,                     # 랜덤 시드
    "weight_light": 0.4,            # Light 가중치
    "weight_hard1": 0.3,            # Hard1 가중치
    "weight_hard2": 0.3,            # Hard2 가중치
    "weight_x": 0.34,               # X축 가중치
    "weight_y": 0.33,               # Y축 가중치
    "weight_z": 0.33,               # Z축 가중치
    "target_suspect_recall": 0.90,  # suspect 임계값 목표 Recall
    "target_break_precision": 0.90, # break 임계값 목표 Precision
    "target_binary_alert_f1": 0.75,     # Alert 이진 분류 F1 목표
    "target_binary_break_f1": 0.70,     # Break 이진 분류 F1 목표
    "target_binary_break_auc": 0.80,    # Break 이진 분류 AUC 목표
    "target_pass_mode": "all_metrics",  # all_metrics | f1_only
}

CURRENT_DIR = Path(__file__).resolve().parent
RUN_BASE_DIR = CURRENT_DIR / "2. mlp_models"
BEST_ALIAS_DIR = CURRENT_DIR / "best_model"
TRAIN_DATA_BASE = CURRENT_DIR / "1. mlp_train_data"
FINAL_BEST_MODEL_DIR = CURRENT_DIR.parent / "best_model"  # 최종 베스트 모델 디렉터리
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
        print("[정보] GPU 사용:", [g.name for g in _gpus])
    except RuntimeError as e:
        print("[오류] GPU 설정 실패:", e)
else:
    print("[정보] GPU 없음, CPU 사용")


def get_latest_train_data_run(base: Path) -> Path:
    """최신 MLP 학습 데이터 run 디렉터리를 찾는다."""
    if not base.exists():
        raise FileNotFoundError(f"Train data base not found: {base}")
    
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        required_files = ["features.npy", "labels.npy", "X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]
        if all((d / f).exists() for f in required_files):
            candidates.append(d)
    
    if not candidates:
        raise FileNotFoundError(f"No valid train data run found in: {base}")
    
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_train_data(train_data_dir: Path) -> Dict:
    """MLP 학습 데이터를 로드한다."""
    print(f"[정보] 학습 데이터 로드: {train_data_dir}")
    X_train = np.load(train_data_dir / "X_train.npy")
    X_val = np.load(train_data_dir / "X_val.npy")
    y_train = np.load(train_data_dir / "y_train.npy")
    y_val = np.load(train_data_dir / "y_val.npy")
    with open(train_data_dir / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(train_data_dir / "feature_names.json", 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    
    print(f"[정보] 데이터 로드 완료: X_train={X_train.shape}, X_val={X_val.shape}")
    print(f"[정보] 피처 목록: {feature_names}")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "metadata": metadata,
        "feature_names": feature_names
    }


def create_mlp_model(hidden_layers: List[int], alpha: float, max_iter: int, 
                     early_stopping: bool, validation_fraction: float, 
                     n_iter_no_change: int, seed: int) -> Pipeline:
    """MLP 모델을 생성한다."""
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
            verbose=False
        ))
    ])
    return mlp


def evaluate_mlp_model(mlp: Pipeline, X_val: np.ndarray, y_val: np.ndarray,
                       target_suspect_recall: float, target_break_precision: float) -> Dict:
    """MLP 모델을 평가한다."""
    val_prob = mlp.predict_proba(X_val)[:, 1]
    suspect_rule = find_threshold_for_recall(y_val, val_prob, target_suspect_recall)
    break_rule = find_threshold_for_precision(y_val, val_prob, target_break_precision)
    
    suspect_th = float(suspect_rule["threshold"])
    break_th = float(break_rule["threshold"])
    
    if break_th < suspect_th:
        break_th = min(1.0, suspect_th + 1e-6)
    pred_alert = (val_prob >= suspect_th).astype(np.int32)
    pred_break = (val_prob >= break_th).astype(np.int32)
    metrics = {
        "binary_alert": {
            "precision": float(precision_score(y_val, pred_alert, zero_division=0)),
            "recall": float(recall_score(y_val, pred_alert, zero_division=0)),
            "f1": float(f1_score(y_val, pred_alert, zero_division=0)),
        },
        "binary_break": {
            "precision": float(precision_score(y_val, pred_break, zero_division=0)),
            "recall": float(recall_score(y_val, pred_break, zero_division=0)),
            "f1": float(f1_score(y_val, pred_break, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_prob)),
        },
        "thresholds": {
            "suspect_threshold": suspect_th,
            "break_threshold": break_th,
            "suspect_rule": suspect_rule,
            "break_rule": break_rule,
        }
    }
    
    return metrics


def check_model_pass_criteria(metrics: Dict, options: Dict) -> Tuple[bool, str]:
    """모델이 통과 기준을 만족하는지 확인한다."""
    alert_f1 = metrics["binary_alert"]["f1"]
    break_f1 = metrics["binary_break"]["f1"]
    break_auc = metrics["binary_break"]["roc_auc"]
    
    target_alert_f1 = options["target_binary_alert_f1"]
    target_break_f1 = options["target_binary_break_f1"]
    target_break_auc = options["target_binary_break_auc"]
    pass_mode = options["target_pass_mode"]
    
    if pass_mode == "all_metrics":
        conditions = [
            (alert_f1 >= target_alert_f1, f"Alert F1: {alert_f1:.4f} >= {target_alert_f1}"),
            (break_f1 >= target_break_f1, f"Break F1: {break_f1:.4f} >= {target_break_f1}"),
            (break_auc >= target_break_auc, f"Break AUC: {break_auc:.4f} >= {target_break_auc}"),
        ]
        passed_conditions = [cond for cond, _ in conditions if cond]
        all_passed = len(passed_conditions) == len(conditions)
        
        feedback = "Pass criteria check:\n"
        for cond, desc in conditions:
            status = "PASS" if cond else "FAIL"
            feedback += f"  [{status}] {desc}\n"
        
        return all_passed, feedback.strip()
    
    elif pass_mode == "f1_only":
        condition = break_f1 >= target_break_f1
        feedback = f"Break F1: {break_f1:.4f} >= {target_break_f1} ({'PASS' if condition else 'FAIL'})"
        return condition, feedback
    
    return False, "Unknown pass mode"


def copy_dependent_models_to_final(train_data_metadata: Dict, mlp_run_name: str):
    """의존 모델을 최종 베스트 모델 디렉터리로 복사한다."""
    print("\n[COPY] 의존 모델을 최종 베스트 디렉터리로 복사합니다.")
    FINAL_BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1) Light 모델 복사
        light_info = train_data_metadata.get("light_model_info", {})
        if light_info and "selected" in light_info:
            light_run = light_info["selected"]["model_run"]
            light_base_dir = CURRENT_DIR.parent / "2. make_light_model" / "best_light_model"
            actual_light_dir = None
            for d in light_base_dir.iterdir():
                if d.is_dir() and light_run.startswith(d.name):
                    actual_light_dir = d
                    break
            
            if actual_light_dir:
                light_model_file = actual_light_dir / "checkpoints" / "best.keras"
                if light_model_file.exists():
                    light_dest_dir = FINAL_BEST_MODEL_DIR / "light_model"
                    light_dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(light_model_file, light_dest_dir / "best.keras")
                    print(f"[COPIED] Light 모델: {actual_light_dir.name}")
                else:
                    print(f"[WARNING] Light 모델 파일을 찾을 수 없습니다: {light_model_file}")
            else:
                print(f"[WARNING] Light 모델 디렉터리를 찾을 수 없습니다: {light_run}")
        
        # 2) Hard1 모델 복사
        hard1_info = train_data_metadata.get("hard1_model_info", {})
        if hard1_info and "selected" in hard1_info:
            hard1_run = hard1_info["selected"]["model_run"]
            hard1_source_dir = CURRENT_DIR.parent / "3. make_hard_model" / "best_hard_model_1st" / hard1_run
            hard1_checkpoints_dir = hard1_source_dir / "checkpoints"
            
            if hard1_checkpoints_dir.exists():
                hard1_dest_dir = FINAL_BEST_MODEL_DIR / "hard1_model"
                hard1_dest_dir.mkdir(parents=True, exist_ok=True)
                for axis in ['x', 'y', 'z']:
                    model_file = hard1_checkpoints_dir / f"conf_{axis}.keras"
                    if model_file.exists():
                        shutil.copy2(model_file, hard1_dest_dir / f"conf_{axis}.keras")
                
                print(f"[COPIED] Hard1 모델: {hard1_run}")
            else:
                print(f"[WARNING] Hard1 체크포인트 디렉터리를 찾을 수 없습니다: {hard1_checkpoints_dir}")
        
        # 3) Hard2 모델 복사
        hard2_info = train_data_metadata.get("hard2_model_info", {})
        if hard2_info and "selected" in hard2_info:
            hard2_run = hard2_info["selected"]["model_run"]
            hard2_base_dir = CURRENT_DIR.parent / "3. make_hard_model" / "best_hard_model_2nd"
            actual_hard2_dir = None
            for d in hard2_base_dir.iterdir():
                if d.is_dir() and hard2_run.startswith(d.name):
                    actual_hard2_dir = d
                    break
            
            if actual_hard2_dir:
                hard2_checkpoints_dir = actual_hard2_dir / "checkpoints"
                if hard2_checkpoints_dir.exists():
                    hard2_dest_dir = FINAL_BEST_MODEL_DIR / "hard2_model"
                    hard2_dest_dir.mkdir(parents=True, exist_ok=True)
                    for axis in ['x', 'y', 'z']:
                        model_file = hard2_checkpoints_dir / f"conf_{axis}.keras"
                        if model_file.exists():
                            shutil.copy2(model_file, hard2_dest_dir / f"conf_{axis}.keras")
                    
                    print(f"[COPIED] Hard2 모델: {actual_hard2_dir.name}")
                else:
                    print(f"[WARNING] Hard2 체크포인트를 찾을 수 없습니다: {hard2_checkpoints_dir}")
            else:
                print(f"[WARNING] Hard2 모델 디렉터리를 찾을 수 없습니다: {hard2_run}")
        
        # 4) MLP 모델 복사
        mlp_source_file = BEST_ALIAS_DIR / "mlp_pipeline.joblib"
        if mlp_source_file.exists():
            mlp_dest_dir = FINAL_BEST_MODEL_DIR / "mlp_model"
            mlp_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mlp_source_file, mlp_dest_dir / "mlp_pipeline.joblib")
            print(f"[COPIED] MLP 모델: {mlp_run_name}")
        
        dependency_info = {
            "created_at": datetime.datetime.now().isoformat(),
            "mlp_model_run": mlp_run_name,
            "dependent_models": {
                "light_model": light_info.get("selected", {}) if light_info else {},
                "hard1_model": hard1_info.get("selected", {}) if hard1_info else {},
                "hard2_model": hard2_info.get("selected", {}) if hard2_info else {},
            },
            "weights": train_data_metadata.get("weights", {}),
            "note": "총 4개 모델 의존 관계 정보. MLP 모델은 Light/Hard1/Hard2 결과 기반으로 학습됨."
        }
        
        dependency_file = FINAL_BEST_MODEL_DIR / "model_dependency_info.json"
        with open(dependency_file, 'w', encoding='utf-8') as f:
            json.dump(dependency_info, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVED] 모델 의존성 정보 저장: {dependency_file}")
        print("[SUCCESS] 의존 모델을 최종 베스트 모델 디렉터리로 복사했습니다.")
        print(f"          최종 디렉터리: {FINAL_BEST_MODEL_DIR}")
        
    except Exception as e:
        print(f"[ERROR] 의존 모델 복사 중 오류 발생: {e}")


def save_best_model_selection(best_dir: Path, run_name: str, metrics: Dict, 
                             train_data_metadata: Dict, passed: bool, feedback: str):
    """베스트 모델 선택 정보를 저장한다."""
    selection_info = {
        "selected": {
            "model_run": run_name,
            "metrics": {
                "binary_alert_f1": metrics["binary_alert"]["f1"],
                "binary_break_f1": metrics["binary_break"]["f1"],
                "binary_break_auc": metrics["binary_break"]["roc_auc"],
                "suspect_threshold": metrics["thresholds"]["suspect_threshold"],
                "break_threshold": metrics["thresholds"]["break_threshold"],
            },
            "pass": passed,
            "feedback": feedback,
            "train_data_info": {
                "light_model_info": train_data_metadata.get("light_model_info", {}),
                "hard1_model_info": train_data_metadata.get("hard1_model_info", {}),
                "hard2_model_info": train_data_metadata.get("hard2_model_info", {}),
                "weights": train_data_metadata.get("weights", {}),
            },
            "selected_at": datetime.datetime.now().isoformat(),
        }
    }
    
    selection_file = best_dir / "best_model_selection.json"
    with open(selection_file, 'w', encoding='utf-8') as f:
        json.dump(selection_info, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] 베스트 모델 선택 정보 저장: {selection_file}")


def update_best_model(run_dir: Path, metrics: Dict, train_data_metadata: Dict, 
                     passed: bool, feedback: str) -> bool:
    """베스트 모델을 업데이트한다."""
    best_selection_file = BEST_ALIAS_DIR / "best_model_selection.json"
    current_f1 = metrics["binary_break"]["f1"]
    current_auc = metrics["binary_break"]["roc_auc"]
    current_alert_f1 = metrics["binary_alert"]["f1"]
    if best_selection_file.exists():
        with open(best_selection_file, 'r', encoding='utf-8') as f:
            current_best = json.load(f)
        
        current_best_f1 = current_best["selected"]["metrics"]["binary_break_f1"]
        current_best_auc = current_best["selected"]["metrics"]["binary_break_auc"]
        current_best_alert_f1 = current_best["selected"]["metrics"]["binary_alert_f1"]
        
        if current_f1 <= current_best_f1:
            print("[INFO] 현재 모델 성능이 기존 베스트보다 낮아 교체하지 않습니다.")
            print(f"       현재 Break F1: {current_f1:.4f} <= 기존 Break F1: {current_best_f1:.4f}")
            return False
        else:
            print("[SUCCESS] 성능 향상으로 베스트 모델을 교체합니다.")
            print(f"         Break F1: {current_best_f1:.4f} -> {current_f1:.4f} (+{current_f1-current_best_f1:.4f})")
            print(f"         Break AUC: {current_best_auc:.4f} -> {current_auc:.4f} (+{current_auc-current_best_auc:.4f})")
            print(f"         Alert F1: {current_best_alert_f1:.4f} -> {current_alert_f1:.4f} (+{current_alert_f1-current_best_alert_f1:.4f})")
    else:
        print("[SUCCESS] 최초 베스트 모델을 생성합니다.")
        print(f"         Break F1: {current_f1:.4f}")
        print(f"         Break AUC: {current_auc:.4f}")
        print(f"         Alert F1: {current_alert_f1:.4f}")
    BEST_ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    if (run_dir / "mlp_pipeline.joblib").exists():
        shutil.copy2(run_dir / "mlp_pipeline.joblib", BEST_ALIAS_DIR / "mlp_pipeline.joblib")
        print(f"[SAVED] 베스트 모델 파일 저장: {BEST_ALIAS_DIR / 'mlp_pipeline.joblib'}")
    
    save_best_model_selection(
        BEST_ALIAS_DIR,
        run_dir.name,
        metrics,
        train_data_metadata,
        passed,
        feedback,
    )
    copy_dependent_models_to_final(train_data_metadata, run_dir.name)
    
    model_update_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": "model_updated",
        "previous_f1": current_best_f1 if best_selection_file.exists() else None,
        "new_f1": current_f1,
        "improvement": current_f1 - current_best_f1 if best_selection_file.exists() else current_f1,
        "new_model_run": run_dir.name,
        "dependent_models": {
            "light": train_data_metadata.get("light_model_info", {}).get("selected", {}).get("model_run", ""),
            "hard1": train_data_metadata.get("hard1_model_info", {}).get("selected", {}).get("model_run", ""),
            "hard2": train_data_metadata.get("hard2_model_info", {}).get("selected", {}).get("model_run", ""),
        }
    }
    log_file = FINAL_BEST_MODEL_DIR / "model_update_history.jsonl"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(model_update_log, ensure_ascii=False) + '\n')
    
    print(f"[LOGGED] 모델 교체 이력 저장: {log_file}")
    
    return True


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
    parser = argparse.ArgumentParser(description="MLP 최종 의사결정 모델 학습")
    parser.add_argument("--train-data-run", default=None, help="MLP train data run name (default: latest)")
    parser.add_argument("--run-name", default=None, help="output run name")
    
    args = parser.parse_args()
    options = USER_OPTIONS.copy()
    
    print("=== MLP 학습 설정 ===")
    print(f"Hidden layers: {options['hidden_layers']}")
    print(f"Alpha (L2): {options['alpha']}")
    print(f"Max iterations: {options['max_iter']}")
    print(f"Target Alert F1: {options['target_binary_alert_f1']}")
    print(f"Target Break F1: {options['target_binary_break_f1']}")
    print(f"Target Break AUC: {options['target_binary_break_auc']}")
    print("=====================\n")
    np.random.seed(options["seed"])
    tf.random.set_seed(options["seed"])
    if args.train_data_run:
        train_data_dir = TRAIN_DATA_BASE / args.train_data_run
    else:
        train_data_dir = get_latest_train_data_run(TRAIN_DATA_BASE)
    
    train_data = load_train_data(train_data_dir)
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_BASE_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[정보] 학습 run 디렉터리: {run_dir}")
    print("[정보] MLP 모델 학습을 시작합니다...")
    mlp = create_mlp_model(
        hidden_layers=options["hidden_layers"],
        alpha=options["alpha"],
        max_iter=options["max_iter"],
        early_stopping=options["early_stopping"],
        validation_fraction=options["validation_fraction"],
        n_iter_no_change=options["n_iter_no_change"],
        seed=options["seed"]
    )
    mlp.fit(train_data["X_train"], train_data["y_train"])
    print("[정보] 학습 완료")
    print("[정보] 모델 평가를 수행합니다...")
    metrics = evaluate_mlp_model(
        mlp, train_data["X_val"], train_data["y_val"],
        options["target_suspect_recall"], options["target_break_precision"]
    )
    passed, feedback = check_model_pass_criteria(metrics, options)
    
    print("\n=== 모델 평가 결과 ===")
    print(f"Binary Alert F1: {metrics['binary_alert']['f1']:.4f}")
    print(f"Binary Break F1: {metrics['binary_break']['f1']:.4f}")
    print(f"Binary Break AUC: {metrics['binary_break']['roc_auc']:.4f}")
    print(f"Suspect Threshold: {metrics['thresholds']['suspect_threshold']:.4f}")
    print(f"Break Threshold: {metrics['thresholds']['break_threshold']:.4f}")
    print(f"\n통과 여부: {'PASS' if passed else 'FAIL'}")
    print(f"피드백:\n{feedback}")
    print("====================\n")
    
    results = {
        "run_name": run_name,
        "created_at": datetime.datetime.now().isoformat(),
        "train_data_run": train_data_dir.name,
        "model_config": {
            "hidden_layers": options["hidden_layers"],
            "alpha": options["alpha"],
            "max_iter": options["max_iter"],
            "early_stopping": options["early_stopping"],
        },
        "metrics": metrics,
        "passed": passed,
        "feedback": feedback,
        "train_data_metadata": train_data["metadata"],
    }
    
    with open(run_dir / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "="*60)
    print("[정보] 베스트 모델 업데이트 여부를 확인합니다.")
    print("="*60)
    
    updated = update_best_model(run_dir, metrics, train_data["metadata"], passed, feedback)
    
    print("="*60)
    if updated:
        print("[정보] 베스트 모델 업데이트 완료")
        print("[정보] 최종 베스트 모델 세트를 main/best_model에 저장했습니다.")
    else:
        print("[정보] 기존 베스트 모델을 유지합니다.")
    print("="*60)
    
    print(f"\n[정보] 학습 완료: {run_dir}")
    print(f"[정보] 로컬 베스트: {BEST_ALIAS_DIR}")
    print(f"[정보] 최종 베스트: {FINAL_BEST_MODEL_DIR}")
    if updated and FINAL_BEST_MODEL_DIR.exists():
        print(f"\n[정보] 최종 베스트 모델 구조:")
        print(f"  {FINAL_BEST_MODEL_DIR}/")
        for model_dir in ["light_model", "hard1_model", "hard2_model", "mlp_model"]:
            model_path = FINAL_BEST_MODEL_DIR / model_dir
            if model_path.exists():
                print(f"  - {model_dir}/")
                for file in model_path.iterdir():
                    if file.is_file():
                        print(f"    - {file.name}")
        
        dependency_file = FINAL_BEST_MODEL_DIR / "model_dependency_info.json"
        if dependency_file.exists():
            print("  - model_dependency_info.json")


if __name__ == "__main__":
    main()
