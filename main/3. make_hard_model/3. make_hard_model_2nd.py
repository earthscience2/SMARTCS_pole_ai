# -*- coding: utf-8 -*-
"""Hard model stage-2(conf head) training with integrated evaluation and best-model selection.

실행 방법:
- 기본: python 3. make_hard_model_2nd.py
  -> 먼저 WSL2에서 GPU 스크립트 시도, 기본은 여기서 종료(재실행 없음)
     (필요 시 HARD_WSL_FALLBACK_CPU=1 로 CPU fallback 허용)
- CPU 강제: python 3. make_hard_model_2nd.py --cpu
  -> GPU 스크립트 건너뛰고 바로 CPU로 실행
- 로컬 모드: python 3. make_hard_model_2nd.py --local
  -> GPU 스크립트 건너뛰고 바로 CPU로 실행
"""

import os
import sys
import subprocess
import shutil
import datetime
import argparse
import json

# Windows 인코딩 문제 해결
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
import shlex
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger, log_event

LOGGER = get_logger("train_hard_model_2nd")


def _normalize_log_message(text: str) -> str:
    if not text:
        return text
    if "?" in text and any(ord(ch) > 127 for ch in text):
        ascii_only = "".join(ch if ord(ch) < 128 else " " for ch in text)
        ascii_only = " ".join(ascii_only.split())
        return f"[legacy-text-normalized] {ascii_only}" if ascii_only else "[legacy-text-normalized]"
    return text


def _log_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    text = sep.join(str(a) for a in args)
    normalized = _normalize_log_message(text)
    if not normalized:
        return
    for line in normalized.splitlines() or [""]:
        log_event(LOGGER, "INFO", "GENERAL", line)


print = _log_print

def try_wsl2_gpu_script():
    """WSL2에서 GPU 스크립트 실행을 시도합니다."""
    if sys.platform != "win32":
        return False
    allow_cpu_fallback = os.environ.get("HARD_WSL_FALLBACK_CPU", "0") == "1"

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parents[1]  # 2단계 위로 올라가서 프로젝트 루트
    _sh_path = Path(_script_dir) / "3. make_hard_model_2nd_gpu.sh"
    
    if not _sh_path.exists():
        print(f"GPU 스크립트를 찾을 수 없습니다: {_sh_path}")
        return False
    
    try:
        # WSL2가 설치되어 있는지 확인
        wsl_check = subprocess.run(["wsl", "--status"], 
                                 capture_output=True, text=True, timeout=10)
        if wsl_check.returncode != 0:
            print("WSL2가 설치되어 있지 않거나 실행할 수 없습니다.")
            return False
            
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = ("/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")) if _drive else str(_abs).replace("\\", "/")
        
        print("=" * 60)
        print("WSL2에서 GPU 스크립트 실행을 시도합니다...")
        print(f"스크립트 경로: {_wsl_path}")
        print("=" * 60)
        
        # 현재 스크립트의 모든 인수를 전달 (--local, --cpu 제외)
        filtered_args = [a for a in sys.argv[1:] if a not in ["--local", "--cpu"]]
        arg_str = " ".join(shlex.quote(a) for a in filtered_args)
        cmd = f"bash {shlex.quote(_wsl_path)} {arg_str}".strip()
        
        ret = subprocess.run(["wsl", "bash", "-lc", cmd], cwd=str(_project_root))
        
        if ret.returncode == 0:
            print("=" * 60)
            print("WSL2 GPU 스크립트가 성공적으로 완료되었습니다!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print(f"WSL2 GPU 스크립트가 실패했습니다 (exit code: {ret.returncode})")
            if allow_cpu_fallback:
                print("CPU 모드로 fallback합니다... (HARD_WSL_FALLBACK_CPU=1)")
                print("=" * 60)
                return False
            print("같은 작업을 CPU로 재실행하지 않습니다(기본). 필요 시 --cpu 로 CPU만 실행하거나, HARD_WSL_FALLBACK_CPU=1 로 fallback을 켜세요.")
            print("=" * 60)
            sys.exit(ret.returncode)
            
    except subprocess.TimeoutExpired:
        print("WSL2 상태 확인 시간 초과. CPU 모드로 fallback합니다.")
        return False
    except FileNotFoundError:
        print("WSL2가 설치되어 있지 않습니다. CPU 모드로 fallback합니다.")
        return False
    except Exception as e:
        print(f"WSL2 실행 중 오류 발생: {e}")
        print("CPU 모드로 fallback합니다.")
        return False

# 명령행 옵션 처리
_run_local = "--local" in sys.argv
_force_cpu = "--cpu" in sys.argv

# 명령행 인수 정리
if "--local" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--local"]
if "--cpu" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--cpu"]

# GPU 스크립트 시도 (--local 또는 --cpu 옵션이 없는 경우)
if not _run_local and not _force_cpu:
    if try_wsl2_gpu_script():
        # GPU 스크립트가 성공하면 여기서 종료됨
        pass

# CPU 모드로 계속 진행
if _run_local:
    print("로컬 모드: CPU로 실행합니다")
elif _force_cpu:
    print("CPU 강제 모드: GPU 스크립트를 건너뛰고 CPU로 실행합니다")
else:
    print("CPU 모드로 fallback하여 실행합니다")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

#  ?
BASE_SEED = 42
BATCH = 32
EPOCHS = 220
CONF_LR = 1e-3
P = 3

# ============================================================================
# ???? ( ???? ?)
# ============================================================================
USER_OPTIONS = {
    "epochs": 220,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "target_overall_best_f1": 0.70,
    "target_overall_auc": 0.70,
    "target_overall_separation": 0.20,
    "target_pass_mode": "all_metrics",  # all_metrics | f1_only
}

IOU_POS_THRESHOLD_PER_AXIS = {
    "x": 0.5,
    "y": 0.4,
    "z": 0.5,
}

current_dir = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = Path(current_dir)
FIRST_STAGE_RUN_DIR_DEFAULT = CURRENT_DIR / "best_hard_model_1st"
RUN_BASE_DIR = CURRENT_DIR / "3. hard_models_2nd"
BEST_ALIAS_DIR = CURRENT_DIR / "best_hard_model_2nd"
DATA_FALLBACK_ROOT = CURRENT_DIR / "5. train_data"

first_stage_run_dir = None
ckpt_dir = None

# GPU ?
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
        print("GPU ?:", [g.name for g in _gpus])
    except RuntimeError as e:
        print("GPU ? ?:", e)
else:
    print("GPU ?, CPU ?")


def get_latest_hard_train_dir(base: Path):
    if base is None or not base.exists():
        return None
    try:
        subs = [d for d in base.iterdir() if d.is_dir()]
    except OSError:
        return None
    if not subs:
        return None
    valid = [
        d
        for d in subs
        if (d / "train" / "break_imgs_train.npy").exists()
        and (d / "test" / "break_imgs_test.npy").exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda d: d.name)


def load_data(train_seq: Path, train_lab: Path, test_seq: Path, test_lab: Path):
    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    K = int((y.shape[1] - 1) // 15)
    assert 1 + 15 * K == y.shape[1], f"y.shape[1]={y.shape[1]} not 1+15*K"
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)
    return X, y, X_test, y_test, K


def slice_roi_targets(y, roi_idx: int, K: int):
    bbox_dim = 12 * K
    mask_dim = 3 * K
    bbox_flat = y[:, 1 : 1 + bbox_dim].astype("float32")
    mask_flat = y[:, 1 + bbox_dim : 1 + bbox_dim + mask_dim].astype("float32")
    bbox = bbox_flat.reshape(-1, 3, K, 4)
    mask = mask_flat.reshape(-1, 3, K)
    bbox_r = bbox[:, roi_idx, :, :].reshape(-1, K * 4)
    mask_r = mask[:, roi_idx, :]
    y_reg_roi = np.concatenate([bbox_r, mask_r], axis=1).astype("float32")
    return y[:, 0:1].astype("float32"), y_reg_roi


def to_corners_np(x):
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred, gt, eps=1e-8):
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


def build_conf_model_for_axis(axis: str, base_model: keras.Model, input_shape):
    gap_layer_name = f"resnet18_like_reg_{axis}_gap"
    try:
        gap_layer = base_model.get_layer(gap_layer_name)
    except ValueError as e:
        raise ValueError(f"GAP ??? ? ??? {gap_layer_name}") from e

    feature_model = keras.Model(base_model.input, gap_layer.output, name=f"{axis}_feature_model")
    feature_model.trainable = False

    inp = keras.Input(shape=input_shape, name=f"{axis}_input_for_conf")
    feat = feature_model(inp)
    x = layers.Dense(128, activation="relu", name=f"{axis}_conf_fc1")(feat)
    x = layers.Dense(64, activation="relu", name=f"{axis}_conf_fc2")(x)
    conf_out = layers.Dense(P, activation="sigmoid", name=f"{axis}_conf_out")(x)

    conf_model = keras.Model(inp, conf_out, name=f"{axis}_conf_model")
    conf_model.compile(
        optimizer=keras.optimizers.Adam(CONF_LR),
        loss="mse",
        metrics=["mae"],
    )
    return conf_model


def compute_conf_targets_for_axis(model: keras.Model, X, y, axis: str, roi_idx: int, K: int):
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    gt_bbox = y_reg[:, : 4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K : 5 * K].reshape(-1, K)

    pred = model.predict(X, batch_size=BATCH, verbose=0)
    pred_full = pred.reshape(-1, P, 5)
    pred_bbox = pred_full[:, :, :4]

    iou = iou_matrix_np(pred_bbox, gt_bbox)
    iou = np.where(gt_mask[:, None, :] > 0.5, iou, 0.0)
    raw_iou_max = iou.max(axis=2)

    thr = IOU_POS_THRESHOLD_PER_AXIS.get(axis, 0.5)
    scaled = (raw_iou_max - thr) / max(1e-6, (1.0 - thr))
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype("float32")


def train_conf_for_axis(axis: str, roi_idx: int, X, y, K: int, conf_ckpt_dir: Path):
    print(f"\n===== Axis {axis.upper()} (ROI {roi_idx}) =====")

    best_path = ckpt_dir / f"best_{axis}.keras"
    if not best_path.exists():
        raise FileNotFoundError(f"best  ?: {best_path}")

    print("Loading base model:", best_path)
    base_model = keras.models.load_model(str(best_path), compile=False)
    base_model.trainable = False

    labels = y[:, 0].astype(int)
    mask_break = labels == 1
    if not np.any(mask_break):
        raise RuntimeError("break(label=1) ???? conf ???????")

    X_break = X[mask_break]
    y_break = y[mask_break]
    print(f"break ?: {X_break.shape[0]} / ? {X.shape[0]}")

    print(f"IoU 湲곕컲 confidence target 怨꾩궛(axis={axis})...")
    y_conf_full = compute_conf_targets_for_axis(base_model, X_break, y_break, axis=axis, roi_idx=roi_idx, K=K)

    indices = np.arange(X_break.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=BASE_SEED,
        stratify=y_break[:, 0].astype(int),
    )

    X_train, X_val = X_break[train_idx], X_break[val_idx]
    yconf_train, yconf_val = y_conf_full[train_idx], y_conf_full[val_idx]

    conf_model = build_conf_model_for_axis(axis, base_model, input_shape=X.shape[1:])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
    ]

    history = conf_model.fit(
        X_train,
        yconf_train,
        validation_data=(X_val, yconf_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    conf_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = conf_ckpt_dir / f"conf_{axis}.keras"
    conf_model.save(str(out_path))
    print(f"Saved conf model for axis {axis} -> {out_path}")

    return history


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(v):
    return round(float(v), 10)


def _build_hard2_signature(
    data_run_name: str,
    first_stage_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    target_f1: float,
    target_auc: float,
    target_sep: float,
    pass_mode: str,
):
    return {
        "data_run": str(data_run_name),
        "first_stage_run": str(first_stage_name),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": _round_float(learning_rate),
        "target_overall_best_f1": _round_float(target_f1),
        "target_overall_auc": _round_float(target_auc),
        "target_overall_separation": _round_float(target_sep),
        "target_pass_mode": str(pass_mode),
    }


def _extract_hard2_signature_from_config(cfg: Dict):
    try:
        training = cfg.get("training", {})
        optimizer = cfg.get("optimizer", {})
        eval_cfg = cfg.get("evaluation_target", {})
        return {
            "data_run": str(cfg.get("data_run", "unknown")),
            "first_stage_run": str(cfg.get("first_stage_run")),
            "epochs": int(training.get("epochs")),
            "batch_size": int(training.get("batch_size")),
            "learning_rate": _round_float(optimizer.get("learning_rate")),
            "target_overall_best_f1": _round_float(eval_cfg.get("target_overall_best_f1")),
            "target_overall_auc": _round_float(eval_cfg.get("target_overall_auc")),
            "target_overall_separation": _round_float(eval_cfg.get("target_overall_separation")),
            "target_pass_mode": str(eval_cfg.get("target_pass_mode")),
        }
    except Exception:
        return None


def _find_duplicate_hard2_run(models_base: Path, signature: Dict):
    if not models_base.exists():
        return None
    for run_dir in sorted([d for d in models_base.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True):
        cfg_path = run_dir / "training_config.json"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        old_sig = _extract_hard2_signature_from_config(cfg)
        if old_sig is not None and old_sig == signature:
            return run_dir.name
    return None


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray):
    auc = None
    best_f1 = None
    best_th = None
    best_pr = None
    best_rc = None

    if np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, scores))
        best_f1, best_th = -1.0, 0.5
        best_pr, best_rc = 0.0, 0.0
        for th in np.linspace(0.0, 1.0, 101):
            pred = (scores >= th).astype(int)
            f1v = float(f1_score(y_true, pred, zero_division=0))
            if f1v >= best_f1:
                best_f1 = f1v
                best_th = float(th)
                best_pr = float(precision_score(y_true, pred, zero_division=0))
                best_rc = float(recall_score(y_true, pred, zero_division=0))
    return {
        "auc_high_iou": auc,
        "best_f1": best_f1 if best_f1 is not None and best_f1 >= 0 else None,
        "best_f1_threshold": best_th,
        "best_f1_precision": best_pr,
        "best_f1_recall": best_rc,
    }


def _compute_axis_eval(axis: str, roi_idx: int, X_test, y_test, K: int, conf_model: keras.Model, base_model: keras.Model):
    labels = y_test[:, 0].astype(int)
    n_total = int(len(labels))
    n_break = int((labels == 1).sum())
    n_normal = int((labels == 0).sum())

    pred_conf = conf_model.predict(X_test, batch_size=BATCH, verbose=0)
    pred_score = pred_conf.max(axis=1).astype(np.float32)

    target_score = np.zeros((n_total,), dtype=np.float32)
    break_mask = labels == 1
    if np.any(break_mask):
        X_break = X_test[break_mask]
        y_break = y_test[break_mask]
        target_break = compute_conf_targets_for_axis(base_model, X_break, y_break, axis=axis, roi_idx=roi_idx, K=K)
        target_score[break_mask] = target_break.max(axis=1)

    binary = (target_score >= 0.5).astype(int)
    cls = _binary_metrics(binary, pred_score)

    conf_break = pred_score[labels == 1]
    conf_normal = pred_score[labels == 0]

    mean_break = float(conf_break.mean()) if conf_break.size else None
    mean_normal = float(conf_normal.mean()) if conf_normal.size else None
    std_break = float(conf_break.std()) if conf_break.size else None
    std_normal = float(conf_normal.std()) if conf_normal.size else None
    separation = None
    if mean_break is not None and mean_normal is not None and std_break is not None and std_normal is not None:
        separation = float((mean_break - mean_normal) / (std_break + std_normal + 1e-8))

    pearson_r = None
    if np.any(break_mask):
        tb = target_score[break_mask]
        pb = pred_score[break_mask]
        if np.std(tb) > 1e-8 and np.std(pb) > 1e-8:
            pearson_r = float(np.corrcoef(tb, pb)[0, 1])

    out = {
        "axis": axis,
        "n_pairs": int(n_total),
        "n_break": n_break,
        "n_normal": n_normal,
        "mean_conf_break": mean_break,
        "mean_conf_normal": mean_normal,
        "separation": separation,
        "pearson_r": pearson_r,
        **cls,
    }
    return out, target_score, pred_score, labels


def _run_stage2_evaluation(
    run_dir: Path,
    target_f1: float,
    target_auc: float,
    target_sep: float,
    pass_mode: str,
    X_test,
    y_test,
    K: int,
):
    local_eval_dir = run_dir / "evaluate"
    local_eval_dir.mkdir(parents=True, exist_ok=True)

    by_axis = {}
    all_targets = []
    all_scores = []
    all_labels = []

    for axis, roi_idx in (("x", 0), ("y", 1), ("z", 2)):
        base_model = keras.models.load_model(str(ckpt_dir / f"best_{axis}.keras"), compile=False)
        conf_model = keras.models.load_model(str(run_dir / "checkpoints" / f"conf_{axis}.keras"), compile=False)
        axis_metrics, target_score, pred_score, labels = _compute_axis_eval(
            axis=axis,
            roi_idx=roi_idx,
            X_test=X_test,
            y_test=y_test,
            K=K,
            conf_model=conf_model,
            base_model=base_model,
        )
        by_axis[axis] = axis_metrics
        all_targets.append(target_score)
        all_scores.append(pred_score)
        all_labels.append(labels)

    all_targets_np = np.concatenate(all_targets, axis=0)
    all_scores_np = np.concatenate(all_scores, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    overall_binary = (all_targets_np >= 0.5).astype(int)
    overall_cls = _binary_metrics(overall_binary, all_scores_np)

    overall_break = all_scores_np[all_labels_np == 1]
    overall_normal = all_scores_np[all_labels_np == 0]
    mean_break = float(overall_break.mean()) if overall_break.size else None
    mean_normal = float(overall_normal.mean()) if overall_normal.size else None
    std_break = float(overall_break.std()) if overall_break.size else None
    std_normal = float(overall_normal.std()) if overall_normal.size else None
    overall_sep = None
    if mean_break is not None and mean_normal is not None and std_break is not None and std_normal is not None:
        overall_sep = float((mean_break - mean_normal) / (std_break + std_normal + 1e-8))

    overall = {
        "n_pairs": int(all_scores_np.shape[0]),
        "mean_conf_break": mean_break,
        "mean_conf_normal": mean_normal,
        "separation": overall_sep,
        **overall_cls,
    }

    evaluation_metrics = {
        "second_stage_run": run_dir.name,
        "first_stage_run": first_stage_run_dir.name,
        "by_axis": by_axis,
        "overall": overall,
    }
    with open(local_eval_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=2)

    overall_f1 = _to_float(overall.get("best_f1"))
    overall_auc = _to_float(overall.get("auc_high_iou"))
    overall_sep_val = _to_float(overall.get("separation"))

    if pass_mode == "f1_only":
        passed = (overall_f1 is not None) and (overall_f1 >= target_f1)
    else:
        passed = (
            (overall_f1 is not None and overall_f1 >= target_f1)
            and (overall_auc is not None and overall_auc >= target_auc)
            and (overall_sep_val is not None and overall_sep_val >= target_sep)
        )

    weak_axes = sorted(
        ["x", "y", "z"],
        key=lambda a: (
            by_axis[a].get("best_f1") if by_axis[a].get("best_f1") is not None else -1.0,
            by_axis[a].get("auc_high_iou") if by_axis[a].get("auc_high_iou") is not None else -1.0,
            by_axis[a].get("separation") if by_axis[a].get("separation") is not None else -9999.0,
        ),
    )

    feedback = {
        "stage": "hard_model_2nd",
        "evaluation_dir": str(local_eval_dir),
        "second_stage_run": run_dir.name,
        "first_stage_run": first_stage_run_dir.name,
        "pass": bool(passed),
        "recommended_retrain": bool(not passed),
        "pass_mode": pass_mode,
        "criteria": {
            "target_overall_best_f1": float(target_f1),
            "target_overall_auc": float(target_auc),
            "target_overall_separation": float(target_sep),
        },
        "actual": {
            "overall_best_f1": overall_f1,
            "overall_auc_high_iou": overall_auc,
            "overall_separation": overall_sep_val,
        },
        "weak_axes_order": weak_axes,
    }
    with open(local_eval_dir / "training_feedback.json", "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    return local_eval_dir


def _rank_key_hard2(metrics: Dict[str, float], target_f1: float, target_auc: float, target_sep: float):
    f1 = _to_float(metrics.get("overall_best_f1"))
    auc = _to_float(metrics.get("overall_auc_high_iou"))
    sep = _to_float(metrics.get("overall_separation"))
    meets = int(
        (f1 is not None and f1 >= target_f1)
        and (auc is not None and auc >= target_auc)
        and (sep is not None and sep >= target_sep)
    )
    return (
        meets,
        (f1 - target_f1) if f1 is not None else -1e9,
        (auc - target_auc) if auc is not None else -1e9,
        (sep - target_sep) if sep is not None else -1e9,
        f1 or -1e9,
        auc or -1e9,
        sep or -1e9,
    )


def _extract_candidate_from_hard2_feedback(feedback_path: Path):
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
    except Exception:
        return None

    run = fb.get("second_stage_run")
    actual = fb.get("actual", {})
    metrics = {
        "overall_best_f1": _to_float(actual.get("overall_best_f1")),
        "overall_auc_high_iou": _to_float(actual.get("overall_auc_high_iou")),
        "overall_separation": _to_float(actual.get("overall_separation")),
    }
    if not run or metrics["overall_best_f1"] is None:
        return None
    return {
        "model_run": run,
        "metrics": metrics,
        "feedback_path": str(feedback_path),
    }


def _collect_best_hard2_candidate(models_base: Path, target_f1: float, target_auc: float, target_sep: float):
    feedback_paths = list(models_base.glob("*/evaluate/training_feedback.json"))
    if not feedback_paths:
        return None
    best = None
    for fp in feedback_paths:
        cand = _extract_candidate_from_hard2_feedback(fp)
        if cand is None:
            continue
        if best is None or _rank_key_hard2(cand["metrics"], target_f1, target_auc, target_sep) > _rank_key_hard2(best["metrics"], target_f1, target_auc, target_sep):
            best = cand
    return best


def _load_current_best_hard2_metrics(best_alias_dir: Path):
    meta_path = best_alias_dir / "best_model_selection.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        selected = meta.get("selected", {})
        m = selected.get("metrics", {})
        return {
            "model_run": selected.get("model_run"),
            "metrics": {
                "overall_best_f1": _to_float(m.get("overall_best_f1")),
                "overall_auc_high_iou": _to_float(m.get("overall_auc_high_iou")),
                "overall_separation": _to_float(m.get("overall_separation")),
            },
        }
    except Exception:
        return None


def _append_best_hard2_history(models_base: Path, payload: Dict):
    history_path = models_base / "best_model_selection_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return history_path


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clear_best_alias_dir(best_alias_dir: Path, preserve_files=None):
    preserve = set(preserve_files or [])
    best_alias_dir.mkdir(parents=True, exist_ok=True)
    for child in list(best_alias_dir.iterdir()):
        if child.name in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_best_model_change_details(
    best_alias_dir: Path,
    model_alias: str,
    action: str,
    criteria: dict,
    selected: dict,
    previous: dict,
    history_path: Path,
    reason: str = "",
):
    best_alias_dir.mkdir(parents=True, exist_ok=True)
    details_path = best_alias_dir / "best_model_change_details.log"
    now = datetime.datetime.now().isoformat()

    selected_metrics = (selected or {}).get("metrics", {})
    previous_metrics = (previous or {}).get("metrics", {}) if previous else {}
    metric_keys = sorted(set(selected_metrics.keys()) | set(previous_metrics.keys()))

    lines = [
        "=" * 80,
        f"timestamp: {now}",
        f"model_alias: {model_alias}",
        f"action: {action}",
        f"reason: {reason or '-'}",
        f"history_path: {history_path}",
        f"selected_run: {(selected or {}).get('model_run')}",
        f"previous_run: {(previous or {}).get('model_run') if previous else '-'}",
        f"criteria: {json.dumps(criteria or {}, ensure_ascii=False)}",
        "[metric_change]",
    ]
    for key in metric_keys:
        prev_v = _safe_float(previous_metrics.get(key))
        cur_v = _safe_float(selected_metrics.get(key))
        if prev_v is None and cur_v is None:
            continue
        delta = None if prev_v is None or cur_v is None else cur_v - prev_v
        lines.append(
            f"- {key}: previous={prev_v if prev_v is not None else '-'} "
            f"current={cur_v if cur_v is not None else '-'} "
            f"delta={delta if delta is not None else '-'}"
        )

    with open(details_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        log_event(
            LOGGER,
            "INFO",
            "MODEL_SELECT",
            "베스트 모델 상세 변경 로그 저장",
            path=details_path,
        )
    return details_path


def _update_best_hard2_model(models_base: Path, best_alias_dir: Path, target_f1: float, target_auc: float, target_sep: float):
    candidate = _collect_best_hard2_candidate(models_base, target_f1, target_auc, target_sep)
    history_path = models_base / "best_model_selection_history.jsonl"
    criteria = {
        "target_overall_best_f1": float(target_f1),
        "target_overall_auc": float(target_auc),
        "target_overall_separation": float(target_sep),
    }
    if candidate is None:
        history_path = _append_best_hard2_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "no_candidate",
                "criteria": criteria,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_2nd",
            action="skip_no_candidate",
            criteria=criteria,
            selected={},
            previous=_load_current_best_hard2_metrics(best_alias_dir),
            history_path=history_path,
            reason="평가 후보 없음",
        )
        print(f"best_hard_model_2nd 갱신 건너뜀: 후보가 없습니다. history={history_path}")
        return

    current_best = _load_current_best_hard2_metrics(best_alias_dir)
    should_replace = (
        current_best is None
        or _rank_key_hard2(candidate["metrics"], target_f1, target_auc, target_sep)
        > _rank_key_hard2(current_best["metrics"], target_f1, target_auc, target_sep)
    )

    src_run_dir = models_base / candidate["model_run"]
    if not src_run_dir.exists():
        history_path = _append_best_hard2_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "candidate_run_missing",
                "criteria": criteria,
                "candidate": candidate,
                "previous": current_best,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_2nd",
            action="skip_missing_selected_run_dir",
            criteria=criteria,
            selected=candidate,
            previous=current_best,
            history_path=history_path,
            reason="선정 run 디렉터리가 없음",
        )
        print(f"best_hard_model_2nd 갱신 건너뜀: 선정 run 경로가 없습니다. history={history_path}")
        return

    if should_replace:
        _clear_best_alias_dir(best_alias_dir, preserve_files={"best_model_change_details.log"})

        dst_run_dir = best_alias_dir / src_run_dir.name
        shutil.copytree(src_run_dir, dst_run_dir)

        selection_payload = {
            "updated_at": datetime.datetime.now().isoformat(),
            "criteria": criteria,
            "selected": {
                "model_run": candidate["model_run"],
                "metrics": candidate["metrics"],
                "feedback_path": candidate["feedback_path"],
            },
            "previous": current_best,
        }
        with open(best_alias_dir / "best_model_selection.json", "w", encoding="utf-8") as f:
            json.dump(selection_payload, f, ensure_ascii=False, indent=2)

        history_path = _append_best_hard2_history(
            models_base, {"timestamp": datetime.datetime.now().isoformat(), "action": "replace_best_model", **selection_payload}
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_2nd",
            action="replace_best_model",
            criteria=criteria,
            selected=selection_payload["selected"],
            previous=current_best,
            history_path=history_path,
            reason="후보 모델이 현재 베스트보다 우수",
        )
        print(
            f"best_hard_model_2nd 갱신: run={candidate['model_run']} "
            f"(f1={candidate['metrics']['overall_best_f1']:.4f}, "
            f"auc={candidate['metrics']['overall_auc_high_iou']:.4f}, sep={candidate['metrics']['overall_separation']:.4f}) "
            f"| saved_to: {dst_run_dir} | history: {history_path}"
        )
        print("베스트 모델이 새로 등록되었습니다.")
    else:
        history_path = _append_best_hard2_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "keep_current_best",
                "criteria": criteria,
                "selected": candidate,
                "previous": current_best,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_2nd",
            action="keep_current_best",
            criteria=criteria,
            selected=candidate,
            previous=current_best,
            history_path=history_path,
            reason="현재 베스트 모델 유지",
        )
        print(
            f"best_hard_model_2nd 유지: current={current_best.get('model_run') if current_best else 'None'} "
            f"candidate={candidate['model_run']} | history: {history_path}"
        )
        print("이번 모델은 기존 베스트 모델을 넘지 못했습니다.")


def main():
    global BATCH, EPOCHS, CONF_LR, first_stage_run_dir, ckpt_dir

    parser = argparse.ArgumentParser(description="Hard 2?conf head)  ?")
    parser.add_argument("--first-stage-run-dir", type=str, default=str(FIRST_STAGE_RUN_DIR_DEFAULT), help="1?hard run ??")
    parser.add_argument("--first-stage-model", type=str, default=None, help="? 1? ? (?? 20260305_1327)")
    parser.add_argument("--list-first-stage-models", action="store_true", help="? ? 1?  ")
    parser.add_argument("--batch-size", type=int, default=USER_OPTIONS["batch_size"], help=" ?")
    parser.add_argument("--epochs", type=int, default=USER_OPTIONS["epochs"], help="? epoch")
    parser.add_argument("--learning-rate", type=float, default=USER_OPTIONS["learning_rate"], help="conf head 학습률")
    parser.add_argument("--target-overall-best-f1", type=float, default=USER_OPTIONS["target_overall_best_f1"])
    parser.add_argument("--target-overall-auc", type=float, default=USER_OPTIONS["target_overall_auc"])
    parser.add_argument("--target-overall-separation", type=float, default=USER_OPTIONS["target_overall_separation"])
    parser.add_argument("--target-pass-mode", type=str, choices=["all_metrics", "f1_only"], default=USER_OPTIONS["target_pass_mode"])
    args = parser.parse_args()

    # 1?   ?
    if args.list_first_stage_models:
        first_models_base = CURRENT_DIR / "2. hard_models_1st"
        if first_models_base.exists():
            models = [d.name for d in first_models_base.iterdir() if d.is_dir()]
            models.sort()
            print("? ? 1?:")
            for model in models:
                print(f"  - {model}")
        else:
            print("1? ??? ????:", first_models_base)
        sys.exit(0)

    BATCH = int(args.batch_size)
    EPOCHS = int(args.epochs)
    CONF_LR = float(args.learning_rate)

    # 1?  ?
    if args.first_stage_model:
        # ? 1???? 
        first_models_base = CURRENT_DIR / "2. hard_models_1st"
        first_stage_run_dir = first_models_base / args.first_stage_model
        if not first_stage_run_dir.exists():
            raise FileNotFoundError(f"? 1???? ??? {first_stage_run_dir}")
        print(f"?????1? ?: {args.first_stage_model}")
    else:
        #  best_hard_model_1st ?
        best_dir = Path(args.first_stage_run_dir).resolve()
        if not best_dir.exists():
            raise FileNotFoundError(f"1?hard run ?? ??? {best_dir}")
        
        # best_model_selection.json? ?   
        selection_file = best_dir / "best_model_selection.json"
        if selection_file.exists():
            import json
            with open(selection_file, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)
            actual_model = selection_data["selected"]["model_run"]
            first_models_base = CURRENT_DIR / "2. hard_models_1st"
            first_stage_run_dir = first_models_base / actual_model
            print(f" 1? ?: best_hard_model_1st -> {actual_model}")
        else:
            #  ?? ?? ?
            subdirs = [d for d in best_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                first_stage_run_dir = subdirs[0]
                print(f" 1? ?: best_hard_model_1st -> {subdirs[0].name}")
            else:
                raise FileNotFoundError(f"best_hard_model_1st? ? ?? ????: {best_dir}")
    
    ckpt_dir = first_stage_run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoints ???? ??? {ckpt_dir}")

    print(f"1?Hard run: {first_stage_run_dir}")
    print(f"? ?: batch={BATCH}, epochs={EPOCHS}, lr={CONF_LR}")
    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    hard_data_base = CURRENT_DIR / "1. hard_train_data"
    data_run_dir = get_latest_hard_train_dir(hard_data_base)
    if data_run_dir is not None:
        data_root = data_run_dir
        print("? ??? hard run):", data_run_dir)
    else:
        data_root = DATA_FALLBACK_ROOT
        print("1. hard_train_data run???  5. train_data fallback ?")

    train_seq = data_root / "train" / "break_imgs_train.npy"
    train_lab = data_root / "train" / "break_labels_train.npy"
    test_seq = data_root / "test" / "break_imgs_test.npy"
    test_lab = data_root / "test" / "break_labels_test.npy"

    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(
            f"? ????????: {train_seq}, {train_lab}\n"
            f"? 1. set_hard_train_data.py????hard_train_data/<run>/train,test NPY?????"
        )

    X, y, X_test, y_test, K = load_data(train_seq, train_lab, test_seq, test_lab)
    print("X:", X.shape, "y:", y.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("K (#GT per ROI axis):", K)

    tf.keras.utils.set_random_seed(BASE_SEED)

    data_run_name = str(data_run_dir.name) if data_run_dir is not None else "5. train_data"
    signature = _build_hard2_signature(
        data_run_name=data_run_name,
        first_stage_name=str(first_stage_run_dir.name),
        epochs=EPOCHS,
        batch_size=BATCH,
        learning_rate=CONF_LR,
        target_f1=float(args.target_overall_best_f1),
        target_auc=float(args.target_overall_auc),
        target_sep=float(args.target_overall_separation),
        pass_mode=str(args.target_pass_mode),
    )
    duplicate_run = _find_duplicate_hard2_run(RUN_BASE_DIR, signature)
    if duplicate_run is not None:
        print(
            f" ? ?:  run '{duplicate_run}'? ???? ???? "
            "? ?? ?/??????"
        )
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = RUN_BASE_DIR / timestamp
    if run_dir.exists():
        raise FileExistsError(f"? timestamp run ? ?? ??? {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    print("2?Hard  run ??:", run_dir)

    second_ckpt_dir = run_dir / "checkpoints"
    second_ckpt_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for axis, roi_idx in (("x", 0), ("y", 1), ("z", 2)):
        hist = train_conf_for_axis(axis, roi_idx, X, y, K=K, conf_ckpt_dir=second_ckpt_dir)
        histories[axis] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    training_config = {
        "timestamp": timestamp,
        "second_stage_run": run_dir.name,
        "first_stage_run": first_stage_run_dir.name,
        "data": {
            "train_shape": list(X.shape),
            "test_shape": list(X_test.shape),
            "num_train_samples": int(len(X)),
            "num_test_samples": int(len(X_test)),
            "bbox_K_per_roi": int(K),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "base_seed": BASE_SEED,
        },
        "model": {
            "architecture": "conf head on top of first-stage ResNet18-like bbox",
            "input_shape": list(X.shape[1:]),
            "pred_boxes_per_axis_P": P,
        },
        "loss": {
            "type": "mse (per-box conf IoU)",
            "iou_pos_threshold_per_axis": {k: float(v) for k, v in IOU_POS_THRESHOLD_PER_AXIS.items()},
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": float(CONF_LR),
        },
        "data_run": data_run_name,
        "evaluation_target": {
            "target_overall_best_f1": float(args.target_overall_best_f1),
            "target_overall_auc": float(args.target_overall_auc),
            "target_overall_separation": float(args.target_overall_separation),
            "target_pass_mode": str(args.target_pass_mode),
        },
        "first_stage_ckpt_dir": str(ckpt_dir),
    }
    with open(run_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    with open(run_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"? ?? (??? {run_dir / 'evaluate'})")
    print("=" * 60)
    print("[EVAL 1/7] hard 2?? ?? ?")
    local_eval_dir = _run_stage2_evaluation(
        run_dir=run_dir,
        target_f1=float(args.target_overall_best_f1),
        target_auc=float(args.target_overall_auc),
        target_sep=float(args.target_overall_separation),
        pass_mode=str(args.target_pass_mode),
        X_test=X_test,
        y_test=y_test,
        K=K,
    )
    print("[EVAL 2/7] 평가 결과를 run 폴더 내부에 저장")
    print("[EVAL 3/7] training_feedback/evaluation_metrics 확인")
    print(f"저장: {local_eval_dir / 'training_feedback.json'}")
    print(f"저장: {local_eval_dir / 'evaluation_metrics.json'}")
    print("[EVAL 4/7] 베스트 모델 후보 비교 준비")
    print("[EVAL 5/7] 베스트 모델 비교/선정")
    _update_best_hard2_model(
        models_base=RUN_BASE_DIR,
        best_alias_dir=BEST_ALIAS_DIR,
        target_f1=float(args.target_overall_best_f1),
        target_auc=float(args.target_overall_auc),
        target_sep=float(args.target_overall_separation),
    )
    print("[EVAL 6/7] 히스토리 기록 완료")
    print("[EVAL 7/7] 상세 평가 완료")
    print(f"상세 평가 결과: {local_eval_dir}")


if __name__ == "__main__":
    main()

