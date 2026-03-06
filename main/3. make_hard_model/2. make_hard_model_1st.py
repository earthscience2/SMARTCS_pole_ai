# -*- coding: utf-8 -*-
"""Hard ResNet bbox model stage-1 training (x/y/z axes).

실행 방법:
- 기본: python 2. make_hard_model_1st.py
  -> 먼저 WSL2에서 GPU 스크립트 시도, 기본은 여기서 종료(재실행 없음)
     (필요 시 HARD_WSL_FALLBACK_CPU=1 로 CPU fallback 허용)
- CPU 강제: python 2. make_hard_model_1st.py --cpu
  -> GPU 스크립트 건너뛰고 바로 CPU로 실행
- 로컬 모드: python 2. make_hard_model_1st.py --local
  -> GPU 스크립트 건너뛰고 바로 CPU로 실행
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Windows 인코딩 문제 해결
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # subprocess 인코딩 설정
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        # Windows 환경에서 C.UTF-8이 없을 수 있음
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            # 설정 불가 시 무시
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger, log_event

LOGGER = get_logger("train_hard_model_1st")

def _log_print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    text = sep.join(str(a) for a in args)
    if not text:
        return
    for line in text.splitlines() or [""]:
        log_event(LOGGER, "INFO", "GENERAL", line)


print = _log_print

def try_wsl2_gpu_script():
    """Try running GPU script in WSL2."""
    if sys.platform != "win32":
        return False
    allow_cpu_fallback = os.environ.get("HARD_WSL_FALLBACK_CPU", "0") == "1"

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parents[1]  # project root
    _sh_path = Path(_script_dir) / "2. make_hard_model_1st_gpu.sh"

    if not _sh_path.exists():
        print(f"[오류] GPU 스크립트를 찾을 수 없습니다: {_sh_path}")
        return False

    try:
        wsl_check = subprocess.run(
            ["wsl", "--status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if wsl_check.returncode != 0:
            print("[오류] WSL2를 사용할 수 없습니다.")
            return False

        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = ("/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")) if _drive else str(_abs).replace("\\", "/")

        print("=" * 60)
        print("[정보] WSL2 GPU 스크립트 실행 시도")
        print(f"스크립트 경로: {_wsl_path}")
        print("=" * 60)

        filtered_args = [a for a in sys.argv[1:] if a not in ["--local", "--cpu"]]

        ret = subprocess.run(
            ["wsl", "bash", _wsl_path] + filtered_args,
            cwd=str(_project_root),
        )

        if ret.returncode == 0:
            print("=" * 60)
            print("[정보] WSL2 GPU 스크립트가 정상 완료되었습니다.")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print(f"[오류] WSL2 GPU 스크립트 실행 실패 (exit code: {ret.returncode})")
            if allow_cpu_fallback:
                print("[정보] CPU 모드로 전환합니다. (HARD_WSL_FALLBACK_CPU=1)")
                print("=" * 60)
                return False
            print("[정보] 기본 설정상 CPU 재실행은 하지 않습니다. 필요 시 --cpu 또는 HARD_WSL_FALLBACK_CPU=1을 사용하세요.")
            print("=" * 60)
            sys.exit(ret.returncode)

    except subprocess.TimeoutExpired:
        print("[오류] WSL2 상태 확인 시간 초과로 CPU 모드로 전환합니다.")
        return False
    except FileNotFoundError:
        print("[오류] WSL2가 설치되어 있지 않아 CPU 모드로 전환합니다.")
        return False
    except Exception as e:
        print(f"[오류] WSL2 실행 오류: {e}")
        print("[정보] CPU 모드로 전환합니다.")
        return False

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
    print("[정보] 로컬 모드: WSL 실행 없이 현재 환경에서 진행합니다.")
elif _force_cpu:
    print("[정보] CPU 강제 모드: GPU 스크립트를 건너뜁니다.")
else:
    if sys.platform == "win32":
        print("[정보] CPU 모드로 실행합니다.")
    else:
        print("[정보] 비-Windows 환경: 현재 환경에서 계속 실행합니다. (GPU 가능 시 사용)")

import datetime
import argparse
import json
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow import 시 Windows CUDA PATH 처리.
if sys.platform == "win32":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.exists(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# GPU 확인
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 목록:", [gpu.name for gpu in gpus])
    except RuntimeError as e:
        print("GPU 설정 오류:", e)
else:
    print("GPU 없음, CPU 사용")

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_SEED = 42

# ============================================================================
# 1) 데이터 경로 / 로드
# ============================================================================

# 1. hard_train_data run 우선, 없으면 5. train_data 사용
data_root = Path(current_dir) / "5. train_data"
train_dir = data_root / "train"
test_dir = data_root / "test"
train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"


def get_latest_hard_train_dir(base: Path):
    """base(1. hard_train_data)에서 train/test NPY가 있는 최신 run 디렉터리를 찾습니다."""
    if base is None or not base.exists():
        return None
    try:
        subs = [d for d in base.iterdir() if d.is_dir()]
    except OSError:
        return None
    if not subs:
        return None
    valid = [
        d for d in subs
        if (d / "train" / "break_imgs_train.npy").exists()
        and (d / "test" / "break_imgs_test.npy").exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda d: d.name)


def load_data():
    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    K = int((y.shape[1] - 1) // 15)
    assert 1 + 15 * K == y.shape[1], f"y.shape[1]={y.shape[1]} not 1+15*K"
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)
    return X, y, X_test, y_test, K


# ============================================================================
# 2) Train/Val split
# ============================================================================

def split_train_val(X, y, test_size=0.2, random_state=BASE_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0].astype(int))


# ============================================================================
# 3) ROI 대상 분리 / 변환
# ============================================================================

def slice_roi_targets(y, roi_idx: int, K: int):
    """y: (N, 1+15K). 諛섑솚: y_cls (N,1), y_reg_roi (N, 5K) = [bbox_r, mask_r]."""
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


BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
# 좌/우 및 상/하 flip 증강 옵션 (기본: 사용 안 함)
USE_AUGMENTATION = False


# ============================================================================
# 사용자 옵션 (학습 파라미터)
# ============================================================================

USER_OPTIONS = {
    "core": {
        "epochs": 300,  # 전체 학습 반복 횟수
        "batch_size": 32,  # 한 번에 처리할 샘플 수
        "learning_rate": 1e-3,  # 학습률 (가중치 업데이트 크기)
        "dropout": 0.45,  # 과적합 방지를 위한 드롭아웃 비율 (0.0~1.0)
        "pred_boxes_per_axis": 3,  # 각 축당 예측할 바운딩 박스 개수
    },
    "sub": {
        "use_augmentation": False,
        "conf_weight": 0.0,  # fixed
        "iou_loss_weight": 0.45,
        "anchor_reg_weight": 0.5,
        "target_mean_best_iou": 0.45,
        "target_ratio_iou_0_5": 0.55,
        "target_ratio_iou_0_7": 0.3,
        "target_pass_mode": "average",  # all_axes | average
    },
}


def _augment_flip(img: tf.Tensor, y_reg: tf.Tensor, K: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    좌우(degree) / 상하(height) 뒤집기.
    bbox [hc, hw, dc, dw] 좌표도 함께 변환.
    - Horizontal flip (axis 1): dc -> 1 - dc
    - Vertical flip (axis 0): hc -> 1 - hc
    """
    bbox_dim = K * 4
    bbox = tf.reshape(y_reg[:bbox_dim], (K, 4))  # [hc, hw, dc, dw] per box
    mask = y_reg[bbox_dim:]

    u1 = tf.random.uniform([], 0, 1)
    img = tf.cond(u1 > 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    dc_new = tf.cond(u1 > 0.5, lambda: 1.0 - bbox[:, 2], lambda: bbox[:, 2])
    bbox = tf.concat([bbox[:, :2], tf.reshape(dc_new, (-1, 1)), bbox[:, 3:]], axis=1)

    u2 = tf.random.uniform([], 0, 1)
    img = tf.cond(u2 > 0.5, lambda: tf.image.flip_up_down(img), lambda: img)
    hc_new = tf.cond(u2 > 0.5, lambda: 1.0 - bbox[:, 0], lambda: bbox[:, 0])
    bbox = tf.concat([tf.reshape(hc_new, (-1, 1)), bbox[:, 1:]], axis=1)

    y_reg = tf.concat([tf.reshape(bbox, [-1]), mask], axis=0)
    return img, y_reg


def make_ds_roi(X, y, roi_idx: int, K: int, training: bool, seed: int):
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    ds = tf.data.Dataset.from_tensor_slices((X, y_reg.astype("float32")))
    if training:
        ds = ds.shuffle(min(len(X), 5000), seed=seed, reshuffle_each_iteration=True)
        if USE_AUGMENTATION:
            ds = ds.map(lambda img, lbl: _augment_flip(img, lbl, K), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)


# ============================================================================
# 4) ResNet18-like 모델 (출력 5*P: hc, hw, dc, dw, confidence)
# ============================================================================

def basic_block(x, filters, stride=(1, 1), prefix="bb"):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.ReLU(name=f"{prefix}_relu1")(x)
    x = layers.Conv2D(filters, 3, strides=(1, 1), padding="same", use_bias=False, name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    if shortcut.shape[-1] != filters or stride != (1, 1):
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{prefix}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{prefix}_proj_bn")(shortcut)
    x = layers.Add(name=f"{prefix}_add")([x, shortcut])
    x = layers.ReLU(name=f"{prefix}_out")(x)
    return x


def make_stage(x, filters, blocks, first_stride=(1, 1), prefix="stage"):
    x = basic_block(x, filters, stride=first_stride, prefix=f"{prefix}_b0")
    for i in range(1, blocks):
        x = basic_block(x, filters, stride=(1, 1), prefix=f"{prefix}_b{i}")
    return x


def build_resnet18_like(input_shape, pred_num, name="resnet18_like_reg", dropout=0.3):
    inp = keras.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(64, 7, strides=(2, 1), padding="same", use_bias=False, name=f"{name}_stem_conv")(inp)
    x = layers.BatchNormalization(name=f"{name}_stem_bn")(x)
    x = layers.ReLU(name=f"{name}_stem_relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=(2, 1), padding="same", name=f"{name}_stem_pool")(x)
    x = make_stage(x, 64, blocks=2, first_stride=(1, 1), prefix=f"{name}_s1")
    x = make_stage(x, 128, blocks=2, first_stride=(2, 1), prefix=f"{name}_s2")
    x = make_stage(x, 256, blocks=2, first_stride=(2, 1), prefix=f"{name}_s3")
    x = make_stage(x, 512, blocks=2, first_stride=(2, 1), prefix=f"{name}_s4")
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    reg_out = layers.Dense(5 * pred_num, activation="sigmoid", name="reg")(x)  # 5*P: hc, hw, dc, dw, conf
    return keras.Model(inp, reg_out, name=name)


# ============================================================================
# 5) IoU / Loss / Metric
# ============================================================================

def iou_2d_from_center_width(pred, true, eps=1e-7):
    """pred, true: (..., 4) [hc, hw, dc, dw]. returns IoU (...,)."""
    pred = tf.cast(pred, tf.float32)
    true = tf.cast(true, tf.float32)
    phc, phw, pdc, pdw = tf.unstack(pred, axis=-1)
    thc, thw, tdc, tdw = tf.unstack(true, axis=-1)
    ph1 = phc - 0.5 * phw
    ph2 = phc + 0.5 * phw
    pd1 = pdc - 0.5 * pdw
    pd2 = pdc + 0.5 * pdw
    th1 = thc - 0.5 * thw
    th2 = thc + 0.5 * thw
    td1 = tdc - 0.5 * tdw
    td2 = tdc + 0.5 * tdw
    ih1 = tf.maximum(ph1, th1)
    ih2 = tf.minimum(ph2, th2)
    id1 = tf.maximum(pd1, td1)
    id2 = tf.minimum(pd2, td2)
    inter_h = tf.maximum(0.0, ih2 - ih1)
    inter_d = tf.maximum(0.0, id2 - id1)
    inter = inter_h * inter_d
    area_p = tf.maximum(0.0, ph2 - ph1) * tf.maximum(0.0, pd2 - pd1)
    area_t = tf.maximum(0.0, th2 - th1) * tf.maximum(0.0, td2 - td1)
    union = area_p + area_t - inter
    return inter / (union + eps)


# conf_loss: bbox IoU가 0.3 이상일 때 confidence loss 적용
CONF_WEIGHT = 0.0
IOU_THRESHOLD_FOR_CONF = 0.3

# IoU 관련 가중치
# - IOU_LOSS_WEIGHT: best IoU 손실 (1 - IoU)
# - ANCHOR_REG_WEIGHT: anchor(box) 정규화 + dead anchor 패널티
IOU_LOSS_WEIGHT = 0.4   # (IoU 손실 가중치)
ANCHOR_REG_WEIGHT = 0.5  # (anchor 정규화 가중치)


def huber_bestpair_loss(P: int, K: int, delta=0.05, conf_weight=CONF_WEIGHT, iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF):
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def loss(y_true, y_pred):
        pred = tf.reshape(y_pred, [-1, P, 5])
        pred_bbox = pred[:, :, :4]
        pred_conf = pred[:, :, 4]
        gt_flat = y_true[:, : 4 * K]
        m_flat = y_true[:, 4 * K :]
        gt = tf.reshape(gt_flat, [-1, K, 4])
        m = tf.cast(m_flat > 0.5, tf.float32)
        iou_mat = iou_2d_from_center_width(pred_bbox[:, :, None, :], gt[:, None, :, :])
        neg_inf = tf.constant(-1e9, dtype=iou_mat.dtype)
        iou_masked = tf.where(m[:, None, :] > 0, iou_mat, neg_inf)
        B = tf.shape(pred_bbox)[0]
        flat = tf.reshape(iou_masked, [B, -1])
        arg = tf.argmax(flat, axis=-1, output_type=tf.int32)
        gt_idx = arg % K
        pred_idx = arg // K
        pred_sel = tf.gather(pred_bbox, pred_idx, batch_dims=1)
        gt_sel = tf.gather(gt, gt_idx, batch_dims=1)
        per_dim = huber(gt_sel, pred_sel)
        bbox_loss_per_sample = tf.reduce_mean(per_dim, axis=-1)
        has_gt = tf.cast(tf.reduce_any(m > 0, axis=1), tf.float32)
        denom = tf.reduce_sum(has_gt) + 1e-7
        bbox_loss = tf.reduce_sum(bbox_loss_per_sample * has_gt) / denom

        # best-pair IoU per sample
        best_iou_per_sample = tf.reduce_max(iou_masked, axis=[1, 2])

        # (1) confidence loss: IoU threshold 이상일 때 conf_loss 적용
        conf_ok = tf.cast(
            (best_iou_per_sample >= iou_threshold_for_conf) & (has_gt > 0), tf.float32
        )
        conf_target = tf.one_hot(pred_idx, depth=P, dtype=tf.float32)
        conf_loss_per_sample = bce(conf_target, pred_conf)
        denom_conf = tf.reduce_sum(conf_ok) + 1e-7
        conf_loss = tf.reduce_sum(conf_loss_per_sample * conf_ok) / denom_conf

        # (2) IoU 손실: best IoU 기반 (L_iou = mean(1 - best_iou))
        iou_loss = tf.reduce_sum((1.0 - best_iou_per_sample) * has_gt) / denom

        # (3) anchor(box) 정규화
        #     anchor별로 가장 IoU가 높은 GT에 대해 Huber 적용
        #     dead anchor(box index=0) 패널티 포함
        # m: (B, K) 1/0 mask, iou_mat: (B, P, K)
        # GT 기준으로 anchor별 best IoU GT index 계산
        iou_mat_pos = tf.where(m[:, None, :] > 0, iou_mat, tf.zeros_like(iou_mat))
        anchor_best_gt_idx = tf.argmax(iou_mat_pos, axis=2, output_type=tf.int32)  # (B, P)
        gt_for_anchor = tf.gather(gt, anchor_best_gt_idx, batch_dims=1)  # (B, P, 4)
        # Huber(reduction=NONE)로 (B, P) 손실 계산
        per_anchor_huber = huber(gt_for_anchor, pred_bbox)  # (B, P)
        anchor_reg_per_sample = tf.reduce_mean(per_anchor_huber, axis=1)  # (B,)
        anchor_reg_loss = tf.reduce_sum(anchor_reg_per_sample * has_gt) / denom

        # (4) dead anchor 패널티: anchor max IoU < 0.05
        #     box_index=0도 dead anchor로 처리
        anchor_max_iou = tf.reduce_max(iou_mat_pos, axis=2)  # (B, P)
        dead_anchor_mask = tf.cast(anchor_max_iou < 0.05, tf.float32)
        dead_anchor_penalty_per_sample = tf.reduce_mean(dead_anchor_mask, axis=1)  # (B,)
        dead_anchor_penalty = tf.reduce_sum(dead_anchor_penalty_per_sample * has_gt) / denom

        total_loss = (
            bbox_loss
            + conf_weight * conf_loss
            + IOU_LOSS_WEIGHT * iou_loss
            + ANCHOR_REG_WEIGHT * (anchor_reg_loss + dead_anchor_penalty)
        )
        return total_loss

    return loss


def bbox_iou_metric_maxPK(P: int, K: int, eps=1e-8):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        gt = tf.reshape(y_true[:, : 4 * K], [-1, K, 4])
        m = tf.reshape(y_true[:, 4 * K : 5 * K], [-1, K])
        pred_full = tf.reshape(y_pred, [-1, P, 5])
        pred = pred_full[:, :, :4]
        iou_mat = iou_2d_from_center_width(pred[:, :, None, :], gt[:, None, :, :], eps=eps)
        neg_inf = tf.constant(-1e9, dtype=iou_mat.dtype)
        iou_masked = tf.where(m[:, None, :] > 0.5, iou_mat, neg_inf)
        best_iou = tf.reduce_max(iou_masked, axis=[1, 2])
        has_gt = tf.cast(tf.reduce_any(m > 0.5, axis=1), tf.float32)
        best_iou = tf.where(has_gt > 0, best_iou, tf.zeros_like(best_iou))
        denom = tf.reduce_sum(has_gt) + eps
        return tf.reduce_sum(best_iou * has_gt) / denom

    metric.__name__ = "bbox_iou"
    return metric


# ============================================================================
# 6) 경로 / 콜백
# ============================================================================

run_dir_base = Path(current_dir) / "2. hard_models_1st"
run_dir = run_dir_base / "dummy"  # main()에서 timestamp로 덮어씀
ckpt_dir = run_dir / "checkpoints"
MONITOR = "val_bbox_iou"
MODE = "max"


def make_callbacks(axis: str):
    best_ckpt_path = ckpt_dir / f"best_{axis}.keras"
    log_dir = run_dir / "logs" / axis / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    cb_best = keras.callbacks.ModelCheckpoint(
        filepath=str(best_ckpt_path),
        monitor=MONITOR,
        mode=MODE,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    cb_tb = keras.callbacks.TensorBoard(log_dir=str(log_dir))
    cb_rlr = keras.callbacks.ReduceLROnPlateau(
        monitor=MONITOR, mode=MODE, factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    cb_es = keras.callbacks.EarlyStopping(
        monitor=MONITOR, mode=MODE, patience=50, restore_best_weights=True, verbose=1
    )
    return [cb_best, cb_tb, cb_rlr, cb_es]


def build_and_compile_model(axis: str, input_shape, P: int, K: int, dropout: float, learning_rate: float):
    """
     ResNet18-like 모델 구성
    - dropout, learning_rate 등 하이퍼파라미터 반영.
    """
    m = build_resnet18_like(
        input_shape=input_shape,
        pred_num=P,
        name=f"resnet18_like_reg_{axis}",
        dropout=dropout,
    )
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=huber_bestpair_loss(
            P=P,
            K=K,
            delta=0.05,
            conf_weight=CONF_WEIGHT,
            iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF,
        ),
        metrics=[bbox_iou_metric_maxPK(P=P, K=K)],
        jit_compile=False,
    )
    return m


# ============================================================================
# 7) : eval_bbox_roi_bestpair
# ============================================================================

def to_corners_np(x):
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred, gt, eps=1e-8):
    """pred: (N,P,4), gt: (N,K,4). return iou: (N,P,K)."""
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


def eval_bbox_roi_bestpair(model, X, y, roi_idx: int, K: int, P: int, batch=32, save_dir=None, prefix=""):
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    gt_bbox = y_reg[:, : 4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K : 5 * K].reshape(-1, K)
    has_gt = gt_mask.sum(axis=1) > 0

    pred = model.predict(X, batch_size=batch, verbose=0)
    pred_full = pred.reshape(-1, P, 5)  # 5*P: hc, hw, dc, dw, conf
    pred_bbox = pred_full[:, :, :4]

    iou = iou_matrix_np(pred_bbox, gt_bbox)
    iou_masked = np.where(gt_mask[:, None, :] > 0.5, iou, -1e9)
    flat = iou_masked.reshape(-1, P * K)
    arg = flat.argmax(axis=1)
    best_iou = flat[np.arange(len(flat)), arg]
    pred_idx = arg // K
    gt_idx = arg % K

    best_iou_valid = best_iou[has_gt]
    mean_best_iou = float(best_iou_valid.mean()) if best_iou_valid.size else np.nan

    pred_sel = pred_bbox[np.arange(len(pred_bbox)), pred_idx]
    gt_sel = gt_bbox[np.arange(len(gt_bbox)), gt_idx]
    err = (pred_sel - gt_sel)[has_gt]
    rmse = np.sqrt(np.mean(err ** 2, axis=0)) if err.size else np.array([np.nan] * 4)

    print(f"[ROI {roi_idx}] mean Best-IoU(max(PxK)) = {mean_best_iou:.4f}  (valid samples={has_gt.sum()}/{len(has_gt)})")
    print(f"[ROI {roi_idx}] RMSE(best-pair) hc/hw/dc/dw = {[float(x) for x in rmse]}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(best_iou_valid, bins=30)
    axes[0].set_title(f"ROI {roi_idx} Best-IoU (max(PxK))")
    axes[0].set_xlabel("Best IoU")
    axes[1].scatter(gt_sel[has_gt, 0], pred_sel[has_gt, 0], s=6)
    axes[1].set_title("True vs Pred (hc)")
    axes[2].scatter(gt_sel[has_gt, 2], pred_sel[has_gt, 2], s=6)
    axes[2].set_title("True vs Pred (dc)")
    fig.suptitle(f"ROI {roi_idx} | mean Best-IoU={mean_best_iou:.4f} | RMSE hc/hw/dc/dw={[float(x) for x in rmse]}", y=1.02)
    fig.tight_layout()

    if save_dir is not None:
        out_png = save_dir / f"{prefix}roi{roi_idx}_bestpair_summary.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print("Saved:", out_png)
        df = pd.DataFrame({
            "idx": np.arange(len(best_iou)),
            "has_gt": has_gt.astype(int),
            "best_iou": best_iou,
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "gt_hc": gt_sel[:, 0], "gt_hw": gt_sel[:, 1], "gt_dc": gt_sel[:, 2], "gt_dw": gt_sel[:, 3],
            "pred_hc": pred_sel[:, 0], "pred_hw": pred_sel[:, 1], "pred_dc": pred_sel[:, 2], "pred_dw": pred_sel[:, 3],
        })
        df.to_csv(save_dir / f"{prefix}roi{roi_idx}_bestpair_rows.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)

    return {"mean_best_iou": mean_best_iou, "rmse": rmse, "best_iou": best_iou, "pred_idx": pred_idx, "gt_idx": gt_idx, "has_gt": has_gt}


def load_best_or_current(best_ckpt_path, fallback_model, P: int, K: int):
    if Path(best_ckpt_path).exists():
        print("Loading best model:", best_ckpt_path)
        m = keras.models.load_model(str(best_ckpt_path), compile=False)
        # 컴파일 필요: loss 재설정, conf_weight 반영
        m.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=huber_bestpair_loss(
                P=P,
                K=K,
                delta=0.05,
                conf_weight=CONF_WEIGHT,
                iou_threshold_for_conf=IOU_THRESHOLD_FOR_CONF,
            ),
            metrics=[bbox_iou_metric_maxPK(P=P, K=K)],
            jit_compile=False,
        )
        return m
    return fallback_model


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(v):
    return round(float(v), 10)


def _build_hard1_signature(
    data_run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    pred_boxes_per_axis: int,
    use_augmentation: bool,
    conf_weight: float,
    iou_loss_weight: float,
    anchor_reg_weight: float,
):
    return {
        "data_run": str(data_run_name),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": _round_float(learning_rate),
        "dropout": _round_float(dropout),
        "pred_boxes_per_axis_P": int(pred_boxes_per_axis),
        "use_augmentation": bool(use_augmentation),
        "conf_weight": _round_float(conf_weight),
        "iou_loss_weight": _round_float(iou_loss_weight),
        "anchor_reg_weight": _round_float(anchor_reg_weight),
    }


def _extract_hard1_signature_from_config(cfg: Dict):
    try:
        training = cfg.get("training", {})
        optimizer = cfg.get("optimizer", {})
        model = cfg.get("model", {})
        loss = cfg.get("loss", {})
        aug = cfg.get("augmentation", {})
        return {
            "data_run": str(cfg.get("data_run")),
            "epochs": int(training.get("epochs")),
            "batch_size": int(training.get("batch_size")),
            "learning_rate": _round_float(optimizer.get("learning_rate")),
            "dropout": _round_float(model.get("dropout")),
            "pred_boxes_per_axis_P": int(model.get("pred_boxes_per_axis_P")),
            "use_augmentation": bool(aug.get("use_flip")),
            "conf_weight": _round_float(loss.get("conf_weight")),
            "iou_loss_weight": _round_float(loss.get("iou_loss_weight")),
            "anchor_reg_weight": _round_float(loss.get("anchor_reg_weight")),
        }
    except Exception:
        return None


def _find_duplicate_hard1_run(models_base: Path, signature: Dict):
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
        old_sig = _extract_hard1_signature_from_config(cfg)
        if old_sig is not None and old_sig == signature:
            return run_dir.name
    return None


def _rank_key_hard1(metrics: Dict[str, float], target_mean_iou: float, target_iou05: float, target_iou07: float):
    m = _to_float(metrics.get("avg_mean_best_iou"))
    r05 = _to_float(metrics.get("avg_ratio_iou_0_5"))
    r07 = _to_float(metrics.get("avg_ratio_iou_0_7"))
    meets = int(
        (m is not None and m >= target_mean_iou)
        and (r05 is not None and r05 >= target_iou05)
        and (r07 is not None and r07 >= target_iou07)
    )
    return (
        meets,
        (m - target_mean_iou) if m is not None else -1e9,
        (r05 - target_iou05) if r05 is not None else -1e9,
        (r07 - target_iou07) if r07 is not None else -1e9,
        m or -1e9,
        r05 or -1e9,
        r07 or -1e9,
    )


def _run_stage1_evaluation(
    run_dir: Path,
    axis_results: Dict[str, Dict[str, float]],
    target_mean_iou: float,
    target_iou05: float,
    target_iou07: float,
    pass_mode: str,
):
    local_eval_dir = run_dir / "evaluate"
    local_eval_dir.mkdir(parents=True, exist_ok=True)

    by_axis = {}
    for axis in ("x", "y", "z"):
        r = axis_results.get(axis, {})
        best_iou = np.asarray(r.get("best_iou", []), dtype=np.float32)
        valid = best_iou[np.isfinite(best_iou)]
        mean_iou = float(valid.mean()) if valid.size else None
        ratio05 = float((valid >= 0.5).mean()) if valid.size else None
        ratio07 = float((valid >= 0.7).mean()) if valid.size else None
        by_axis[axis] = {
            "mean_best_iou": mean_iou,
            "ratio_iou_0_5": ratio05,
            "ratio_iou_0_7": ratio07,
            "num_pairs": int(valid.size),
        }

    vals_mean = [v["mean_best_iou"] for v in by_axis.values() if v["mean_best_iou"] is not None]
    vals05 = [v["ratio_iou_0_5"] for v in by_axis.values() if v["ratio_iou_0_5"] is not None]
    vals07 = [v["ratio_iou_0_7"] for v in by_axis.values() if v["ratio_iou_0_7"] is not None]
    avg_mean = float(np.mean(vals_mean)) if vals_mean else None
    avg05 = float(np.mean(vals05)) if vals05 else None
    avg07 = float(np.mean(vals07)) if vals07 else None

    if pass_mode == "all_axes":
        passed = all(
            (by_axis[a]["mean_best_iou"] is not None and by_axis[a]["mean_best_iou"] >= target_mean_iou)
            and (by_axis[a]["ratio_iou_0_5"] is not None and by_axis[a]["ratio_iou_0_5"] >= target_iou05)
            and (by_axis[a]["ratio_iou_0_7"] is not None and by_axis[a]["ratio_iou_0_7"] >= target_iou07)
            for a in ("x", "y", "z")
        )
    else:
        passed = (
            avg_mean is not None and avg05 is not None and avg07 is not None
            and avg_mean >= target_mean_iou
            and avg05 >= target_iou05
            and avg07 >= target_iou07
        )

    metrics = {
        "model_run": run_dir.name,
        "pass_mode": pass_mode,
        "by_axis": by_axis,
        "overall": {
            "avg_mean_best_iou": avg_mean,
            "avg_ratio_iou_0_5": avg05,
            "avg_ratio_iou_0_7": avg07,
        },
    }
    with open(local_eval_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    feedback = {
        "stage": "hard_model_1st",
        "evaluation_dir": str(local_eval_dir),
        "model_run": run_dir.name,
        "pass": bool(passed),
        "recommended_retrain": bool(not passed),
        "pass_mode": pass_mode,
        "criteria": {
            "target_mean_best_iou": float(target_mean_iou),
            "target_ratio_iou_0_5": float(target_iou05),
            "target_ratio_iou_0_7": float(target_iou07),
        },
        "actual": {
            "avg_mean_best_iou": avg_mean,
            "avg_ratio_iou_0_5": avg05,
            "avg_ratio_iou_0_7": avg07,
        },
    }
    with open(local_eval_dir / "training_feedback.json", "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    return local_eval_dir


def _extract_candidate_from_hard1_feedback(feedback_path: Path):
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
    except Exception:
        return None
    run = fb.get("model_run")
    actual = fb.get("actual", {})
    metrics = {
        "avg_mean_best_iou": _to_float(actual.get("avg_mean_best_iou")),
        "avg_ratio_iou_0_5": _to_float(actual.get("avg_ratio_iou_0_5")),
        "avg_ratio_iou_0_7": _to_float(actual.get("avg_ratio_iou_0_7")),
    }
    if not run or metrics["avg_mean_best_iou"] is None:
        return None
    return {
        "model_run": run,
        "metrics": metrics,
        "feedback_path": str(feedback_path),
    }


def _collect_best_hard1_candidate(models_base: Path, target_mean_iou: float, target_iou05: float, target_iou07: float):
    feedback_paths = list(models_base.glob("*/evaluate/training_feedback.json"))
    if not feedback_paths:
        return None
    best = None
    for fp in feedback_paths:
        cand = _extract_candidate_from_hard1_feedback(fp)
        if cand is None:
            continue
        if best is None or _rank_key_hard1(cand["metrics"], target_mean_iou, target_iou05, target_iou07) > _rank_key_hard1(best["metrics"], target_mean_iou, target_iou05, target_iou07):
            best = cand
    return best


def _load_current_best_hard1_metrics(best_alias_dir: Path):
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
                "avg_mean_best_iou": _to_float(m.get("avg_mean_best_iou")),
                "avg_ratio_iou_0_5": _to_float(m.get("avg_ratio_iou_0_5")),
                "avg_ratio_iou_0_7": _to_float(m.get("avg_ratio_iou_0_7")),
            },
        }
    except Exception:
        return None


def _append_best_hard1_history(models_base: Path, record: dict):
    models_base.mkdir(parents=True, exist_ok=True)
    history_path = models_base / "best_model_selection_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
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


def _update_best_hard1_model(models_base: Path, best_alias_dir: Path, target_mean_iou: float, target_iou05: float, target_iou07: float):
    candidate = _collect_best_hard1_candidate(models_base, target_mean_iou, target_iou05, target_iou07)
    criteria = {
        "target_mean_best_iou": float(target_mean_iou),
        "target_ratio_iou_0_5": float(target_iou05),
        "target_ratio_iou_0_7": float(target_iou07),
    }
    if candidate is None:
        history_path = _append_best_hard1_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "skip_no_candidate",
                "criteria": criteria,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_1st",
            action="skip_no_candidate",
            criteria=criteria,
            selected={},
            previous=_load_current_best_hard1_metrics(best_alias_dir),
            history_path=history_path,
            reason="평가 후보 없음",
        )
        print(f"best_hard_model_1st 갱신 건너뜀: 후보가 없습니다. history={history_path}")
        return

    current_best = _load_current_best_hard1_metrics(best_alias_dir)
    should_replace = (
        current_best is None
        or _rank_key_hard1(candidate["metrics"], target_mean_iou, target_iou05, target_iou07)
        > _rank_key_hard1(current_best["metrics"], target_mean_iou, target_iou05, target_iou07)
    )
    selected_run_dir = models_base / candidate["model_run"]
    if not selected_run_dir.exists():
        history_path = _append_best_hard1_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "skip_missing_selected_run_dir",
                "criteria": criteria,
                "selected": candidate,
                "previous": current_best,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_1st",
            action="skip_missing_selected_run_dir",
            criteria=criteria,
            selected=candidate,
            previous=current_best,
            history_path=history_path,
            reason="선정 run 디렉터리가 없음",
        )
        print(f"best_hard_model_1st 갱신 건너뜀: 선정 run 경로가 없습니다. history={history_path}")
        return

    if should_replace:
        _clear_best_alias_dir(best_alias_dir, preserve_files={"best_model_change_details.log"})
        dst_run_dir = best_alias_dir / candidate["model_run"]
        shutil.copytree(selected_run_dir, dst_run_dir)
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
        history_path = _append_best_hard1_history(
            models_base, {"timestamp": datetime.datetime.now().isoformat(), "action": "replace_best_model", **selection_payload}
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_hard_model_1st",
            action="replace_best_model",
            criteria=criteria,
            selected=selection_payload["selected"],
            previous=current_best,
            history_path=history_path,
            reason="후보 모델이 현재 베스트보다 우수",
        )
        print("\n" + "=" * 80)
        print("🎉 베스트 모델 교체 성공!")
        print("=" * 80)
        print(f"✅ 새로운 베스트 모델: {candidate['model_run']}")
        print(f"   - Avg Mean IoU: {candidate['metrics']['avg_mean_best_iou']:.4f}")
        print(f"   - IoU ≥ 0.5: {candidate['metrics']['avg_ratio_iou_0_5']:.4f}")
        print(f"   - IoU ≥ 0.7: {candidate['metrics']['avg_ratio_iou_0_7']:.4f}")
        if current_best:
            print(f"\n📊 이전 베스트 모델: {current_best.get('model_run', 'N/A')}")
            print(f"   - Avg Mean IoU: {current_best['metrics'].get('avg_mean_best_iou', 0):.4f} → {candidate['metrics']['avg_mean_best_iou']:.4f}")
            print(f"   - IoU ≥ 0.5: {current_best['metrics'].get('avg_ratio_iou_0_5', 0):.4f} → {candidate['metrics']['avg_ratio_iou_0_5']:.4f}")
            print(f"   - IoU ≥ 0.7: {current_best['metrics'].get('avg_ratio_iou_0_7', 0):.4f} → {candidate['metrics']['avg_ratio_iou_0_7']:.4f}")
        print(f"\n💾 저장 위치: {dst_run_dir}")
        print(f"📝 히스토리: {history_path}")
        print("=" * 80 + "\n")
    else:
        history_path = _append_best_hard1_history(
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
            model_alias="best_hard_model_1st",
            action="keep_current_best",
            criteria=criteria,
            selected=candidate,
            previous=current_best,
            history_path=history_path,
            reason="현재 베스트 모델 유지",
        )
        print("\n" + "=" * 80)
        print("ℹ️  베스트 모델 유지")
        print("=" * 80)
        print(f"🏆 현재 베스트 모델: {current_best.get('model_run', 'N/A')}")
        print(f"   - Avg Mean IoU: {current_best['metrics'].get('avg_mean_best_iou', 0):.4f}")
        print(f"   - IoU ≥ 0.5: {current_best['metrics'].get('avg_ratio_iou_0_5', 0):.4f}")
        print(f"   - IoU ≥ 0.7: {current_best['metrics'].get('avg_ratio_iou_0_7', 0):.4f}")
        print(f"\n🔍 후보 모델: {candidate['model_run']}")
        print(f"   - Avg Mean IoU: {candidate['metrics']['avg_mean_best_iou']:.4f}")
        print(f"   - IoU ≥ 0.5: {candidate['metrics']['avg_ratio_iou_0_5']:.4f}")
        print(f"   - IoU ≥ 0.7: {candidate['metrics']['avg_ratio_iou_0_7']:.4f}")
        print(f"\n❌ 결과: 후보 모델이 현재 베스트 모델을 넘지 못했습니다.")
        print(f"📝 히스토리: {history_path}")
        print("=" * 80 + "\n")


# ============================================================================
# 8) main
# ============================================================================

def main():
    global run_dir, ckpt_dir, train_dir, test_dir, train_seq, train_lab, test_seq, test_lab
    global BATCH, USE_AUGMENTATION, CONF_WEIGHT, IOU_LOSS_WEIGHT, ANCHOR_REG_WEIGHT

    parser = argparse.ArgumentParser(description="Hard 1차 bbox 모델 학습")
    parser.add_argument("--exp", type=int, default=None, help=argparse.SUPPRESS)  # 내부 실험용
    parser.add_argument("--epochs", type=int, default=USER_OPTIONS["core"]["epochs"], help="학습 epoch")
    parser.add_argument("--batch-size", type=int, default=USER_OPTIONS["core"]["batch_size"], help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=USER_OPTIONS["core"]["learning_rate"], help="학습률")
    parser.add_argument("--dropout", type=float, default=USER_OPTIONS["core"]["dropout"], help="dropout")
    parser.add_argument("--pred-boxes-per-axis", type=int, default=USER_OPTIONS["core"]["pred_boxes_per_axis"], help="축당 예측 박스 수 (P)")
    parser.add_argument("--run-tag", type=str, default="", help="run 이름 suffix")
    args = parser.parse_args()
    # Data path: prefer latest run under 1. hard_train_data, fallback to 5. train_data
    hard_data_base = Path(current_dir) / "1. hard_train_data"
    data_run_dir = get_latest_hard_train_dir(hard_data_base)
    if data_run_dir is not None:
        train_dir = data_run_dir / "train"
        test_dir = data_run_dir / "test"
        train_seq = train_dir / "break_imgs_train.npy"
        train_lab = train_dir / "break_labels_train.npy"
        test_seq = test_dir / "break_imgs_test.npy"
        test_lab = test_dir / "break_labels_test.npy"
        print("사용 데이터 run (hard):", data_run_dir)
    else:
        print("1. hard_train_data run 없음, 5. train_data로 fallback")

    BATCH = int(args.batch_size)
    USE_AUGMENTATION = bool(USER_OPTIONS["sub"]["use_augmentation"])
    CONF_WEIGHT = 0.0
    IOU_LOSS_WEIGHT = float(USER_OPTIONS["sub"]["iou_loss_weight"])
    ANCHOR_REG_WEIGHT = float(USER_OPTIONS["sub"]["anchor_reg_weight"])
    TARGET_MEAN_IOU = float(USER_OPTIONS["sub"]["target_mean_best_iou"])
    TARGET_IOU_05 = float(USER_OPTIONS["sub"]["target_ratio_iou_0_5"])
    TARGET_IOU_07 = float(USER_OPTIONS["sub"]["target_ratio_iou_0_7"])
    TARGET_PASS_MODE = str(USER_OPTIONS["sub"]["target_pass_mode"])

    print("[정보] 학습 설정:")
    print("  P:", int(args.pred_boxes_per_axis), "batch:", BATCH, "use_aug:", USE_AUGMENTATION)
    print("  lr:", float(args.learning_rate), "dropout:", float(args.dropout))
    print("  conf_weight:", CONF_WEIGHT, "(fixed)", "iou_loss_weight:", IOU_LOSS_WEIGHT, "anchor_reg_weight:", ANCHOR_REG_WEIGHT)

    data_run_name = str(data_run_dir.name) if data_run_dir is not None else "5. train_data"
    signature = _build_hard1_signature(
        data_run_name=data_run_name,
        epochs=int(args.epochs),
        batch_size=BATCH,
        learning_rate=float(args.learning_rate),
        dropout=float(args.dropout),
        pred_boxes_per_axis=int(args.pred_boxes_per_axis),
        use_augmentation=USE_AUGMENTATION,
        conf_weight=0.0,
        iou_loss_weight=IOU_LOSS_WEIGHT,
        anchor_reg_weight=ANCHOR_REG_WEIGHT,
    )
    duplicate_run = _find_duplicate_hard1_run(run_dir_base, signature)
    if duplicate_run is not None:
        print("\n" + "=" * 80)
        print("⚠️  중복 실행 방지")
        print("=" * 80)
        print(f"🔍 감지된 중복 run: {duplicate_run}")
        print(f"📋 동일한 학습 설정으로 이미 실행된 run이 존재합니다.")
        print(f"💡 중복 실행을 방지하기 위해 학습을 건너뜁니다.")
        print(f"📂 기존 run 위치: {run_dir_base / duplicate_run}")
        print("=" * 80 + "\n")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if args.run_tag:
        timestamp = f"{timestamp}_{args.run_tag}"
    run_dir = run_dir_base / f"{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("run_dir:", run_dir)
    print("ckpt_dir:", ckpt_dir)

    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(
            f"학습 데이터가 없습니다: {train_seq}, {train_lab}\n"
            f"먼저 1. set_hard_train_data.py로 hard_train_data/<run>/train,test NPY를 생성하세요."
        )

    X, y, X_test, y_test, K = load_data()
    print("X:", X.shape, "y:", y.shape, "cls counts:", np.unique(y[:, 0].astype(int), return_counts=True))
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    X_train, X_val, y_train, y_val = split_train_val(X, y)
    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    P = int(args.pred_boxes_per_axis)
    EPOCHS = int(args.epochs)

    # 학습 설정 기록
    training_config = {
        "timestamp": timestamp,
        "model_run": run_dir.name,
        "data_run": str(data_run_dir.name) if data_run_dir is not None else "5. train_data",
        "data": {
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
            "test_shape": list(X_test.shape),
            "num_train_samples": int(len(X_train)),
            "num_val_samples": int(len(X_val)),
            "num_test_samples": int(len(X_test)),
            "bbox_K_per_roi": int(K),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "base_seed": BASE_SEED,
            "train_val_split": {
                "test_size": 0.2,
                "random_state": BASE_SEED,
                "stratify": True,
            },
        },
        "model": {
            "architecture": "ResNet18-like bbox+confidence",
            "input_shape": list(X_train.shape[1:]),
            "pred_boxes_per_axis_P": P,
            "dropout": float(args.dropout),
        },
        "loss": {
            "type": "Huber_bestpair + BCE(confidence)",
            "huber_delta": 0.05,
            "conf_weight": float(CONF_WEIGHT),
            "iou_threshold_for_conf": float(IOU_THRESHOLD_FOR_CONF),
            "iou_loss_weight": float(IOU_LOSS_WEIGHT),
            "anchor_reg_weight": float(ANCHOR_REG_WEIGHT),
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": float(args.learning_rate),
        },
        "augmentation": {
            "use_flip": bool(USE_AUGMENTATION),
        },
    }
    with open(run_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    print("설정 저장:", run_dir / "training_config.json")

    # fallback 모델(베스트 로딩 실패 시)
    model_x = build_resnet18_like(
        input_shape=X_train.shape[1:],
        pred_num=P,
        name="resnet18_like_reg_x",
        dropout=float(args.dropout),
    )
    model_y = build_resnet18_like(
        input_shape=X_train.shape[1:],
        pred_num=P,
        name="resnet18_like_reg_y",
        dropout=float(args.dropout),
    )
    model_z = build_resnet18_like(
        input_shape=X_train.shape[1:],
        pred_num=P,
        name="resnet18_like_reg_z",
        dropout=float(args.dropout),
    )

    histories = {}
    for axis, roi_idx in [("x", 0), ("y", 1), ("z", 2)]:
        seed = BASE_SEED + roi_idx
        tf.keras.utils.set_random_seed(seed)
        ds_train = make_ds_roi(X_train, y_train, roi_idx=roi_idx, K=K, training=True, seed=seed)
        ds_val = make_ds_roi(X_val, y_val, roi_idx=roi_idx, K=K, training=False, seed=seed)
        model = build_and_compile_model(
            axis,
            X_train.shape[1:],
            P=P,
            K=K,
            dropout=float(args.dropout),
            learning_rate=float(args.learning_rate),
        )
        callbacks = make_callbacks(axis)
        history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        histories[axis] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # 테스트용 seed
    best_x = load_best_or_current(ckpt_dir / "best_x.keras", model_x, P=P, K=K)
    best_y = load_best_or_current(ckpt_dir / "best_y.keras", model_y, P=P, K=K)
    best_z = load_best_or_current(ckpt_dir / "best_z.keras", model_z, P=P, K=K)

    seed_test = BASE_SEED + 1000
    for name, m, roi_idx in [("X", best_x, 0), ("Y", best_y, 1), ("Z", best_z, 2)]:
        ds_t = make_ds_roi(X_test, y_test, roi_idx=roi_idx, K=K, training=False, seed=seed_test)
        res = m.evaluate(ds_t, verbose=1)
        print(f"== Test {name} == ", dict(zip(m.metrics_names, res)))

    out_dir = run_dir / "eval_bestpair"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("평가 결과:", out_dir)
    res_x = eval_bbox_roi_bestpair(best_x, X_test, y_test, roi_idx=0, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="x_")
    res_y = eval_bbox_roi_bestpair(best_y, X_test, y_test, roi_idx=1, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="y_")
    res_z = eval_bbox_roi_bestpair(best_z, X_test, y_test, roi_idx=2, K=K, P=P, batch=BATCH, save_dir=out_dir, prefix="z_")

    with open(run_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    print("학습 완료. 체크포인트:", ckpt_dir)
    print("기본 평가 결과:", out_dir)

    # 상세 평가 + 베스트 모델 선정
    print("\n" + "=" * 60)
    print(f"상세 평가 (저장: {run_dir / 'evaluate'})")
    print("=" * 60)
    print("[정보][평가 1/7] hard 1차 평가 스크립트 실행")
    local_eval_dir = _run_stage1_evaluation(
        run_dir=run_dir,
        axis_results={"x": res_x, "y": res_y, "z": res_z},
        target_mean_iou=TARGET_MEAN_IOU,
        target_iou05=TARGET_IOU_05,
        target_iou07=TARGET_IOU_07,
        pass_mode=TARGET_PASS_MODE,
    )
    print("[정보][평가 2/7] 평가 결과를 run 폴더 내부로 동기화")
    print("[정보][평가 3/7] training_feedback/evaluation_metrics 확인")
    print(f"저장: {local_eval_dir / 'training_feedback.json'}")
    print(f"저장: {local_eval_dir / 'evaluation_metrics.json'}")
    print("[정보][평가 4/7] 베스트 모델 후보 비교 준비")
    print("[정보][평가 5/7] 베스트 모델 비교/선정")
    _update_best_hard1_model(
        models_base=run_dir_base,
        best_alias_dir=Path(current_dir) / "best_hard_model_1st",
        target_mean_iou=TARGET_MEAN_IOU,
        target_iou05=TARGET_IOU_05,
        target_iou07=TARGET_IOU_07,
    )
    print("[정보][평가 6/7] 히스토리 기록 완료")
    print("[정보][평가 7/7] 상세 평가 완료")
    print(f"상세 평가 결과: {local_eval_dir}")


if __name__ == "__main__":
    main()
