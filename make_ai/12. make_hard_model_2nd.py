"""Hard ResNet 모델 2차 학습 - confidence head 추가 학습 (x/y/z축)"""

import os
import sys
import subprocess
from pathlib import Path

import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ============================================================================
# Windows에서 실행 시 WSL2로 넘겨 GPU 사용 (bbox 학습 스크립트와 동일 패턴)
# ============================================================================

_run_local = "--local" in sys.argv or sys.platform != "win32"
if _run_local:
    if "--local" in sys.argv:
        # 이후 로직에서는 --local 플래그 제거
        sys.argv = [a for a in sys.argv if a != "--local"]
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parent
    _sh_path = Path(_script_dir) / "12. make_hard_model_2nd_wsl2.sh"
    if not _sh_path.exists():
        _sh_path = _project_root / "make_ai" / "12. make_hard_model_2nd_wsl2.sh"
    if _sh_path.exists():
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = (
            "/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")
        ) if _drive else str(_abs).replace("\\", "/")
        print("WSL2에서 GPU 학습(Conf head) 실행:", _wsl_path)
        ret = subprocess.run(["wsl", "bash", _wsl_path], cwd=str(_project_root))
        sys.exit(ret.returncode)
    # 스크립트 없으면 로컬(현재 Python)에서 그대로 진행


# ============================================================================
# 설정
# ============================================================================

BASE_SEED = 42
BATCH = 32
EPOCHS = 300
P = 3  # prediction 개수 (기존 ResNet bbox 모델과 동일)

# conf 학습에 사용할 IoU 기준(축별 설정)
# - 연속 타깃(target_conf = max IoU)을 학습하지만,
#   축별로 "고품질 박스"를 판단할 때 참고할 threshold를 함께 기록해 둔다.
IOU_POS_THRESHOLD_PER_AXIS = {
    "x": 0.5,
    "y": 0.4,  # Y축은 상대적으로 IoU 분포가 낮아 약간 더 느슨하게
    "z": 0.5,
}

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1차 Hard 베스트 모델 디렉터리 (x/y/z best_x/y/z.keras 모아둔 곳)
# 예: make_ai/hard_model_1st_best/checkpoints/best_x.keras ...
FIRST_STAGE_RUN_DIR = Path(current_dir) / "hard_model_1st_best"


# 기존에는 10. hard_models_1st에서 최신 run을 고르도록 했으나,
# 이제는 명시적으로 hard_model_1st_best 폴더를 기준으로 사용한다.
first_stage_run_dir = FIRST_STAGE_RUN_DIR
if not first_stage_run_dir.exists():
    raise FileNotFoundError(f"1차 Hard 베스트 모델 디렉터리를 찾을 수 없습니다: {first_stage_run_dir}")

print(f"1차 Hard 모델 run 사용 (hard_model_1st_best): {first_stage_run_dir}")

ckpt_dir = first_stage_run_dir / "checkpoints"
if not ckpt_dir.is_dir():
    raise FileNotFoundError(f"체크포인트 디렉터리 없음: {ckpt_dir}")


# ============================================================================
# 데이터 로드 / ROI 타겟 슬라이스 / IoU 계산 (기존 코드와 동일한 형태)
#  - 기본은 9. hard_train_data 최신 run을 사용, 없으면 5. train_data fallback
# ============================================================================

DATA_FALLBACK_ROOT = Path(current_dir) / "5. train_data"
data_root = DATA_FALLBACK_ROOT
train_dir = data_root / "train"
test_dir = data_root / "test"

train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"


def get_latest_hard_train_dir(base: Path):
    """9. hard_train_data 안에서 train/test NPY가 모두 있는 run 중 이름 기준 최신 폴더 반환."""
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


def load_data():
    X = np.load(train_seq).astype(np.float32)
    y = np.load(train_lab).astype(np.float32)
    K = int((y.shape[1] - 1) // 15)
    assert 1 + 15 * K == y.shape[1], f"y.shape[1]={y.shape[1]} not 1+15*K"
    X_test = np.load(test_seq).astype(np.float32)
    y_test = np.load(test_lab).astype(np.float32)
    return X, y, X_test, y_test, K


def slice_roi_targets(y, roi_idx: int, K: int):
    """
    y: (N, 1+15K).
    반환: y_cls (N,1), y_reg_roi (N, 5K) = [bbox_r(4K), mask_r(K)].
    (기존 train_break_pattern_resnet_bbox*.py 와 동일 로직)
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


def to_corners_np(x):
    hc, hw, dc, dw = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    hmin = hc - 0.5 * hw
    hmax = hc + 0.5 * hw
    dmin = dc - 0.5 * dw
    dmax = dc + 0.5 * dw
    return hmin, hmax, dmin, dmax


def iou_matrix_np(pred, gt, eps=1e-8):
    """
    pred: (N, P, 4), gt: (N, K, 4)  ->  IoU: (N, P, K)
    (기존 eval_bbox_roi_bestpair 와 동일 구조)
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
# Axis별: best 모델 로드 + IoU 기반 confidence 타깃 계산 + conf head 학습
# ============================================================================

def build_conf_model_for_axis(axis: str, base_model: keras.Model, input_shape):
    """
    - base_model: best_x / best_y / best_z (bbox 회귀까지 포함된 전체 모델)
    - backbone(GAP 전까지)은 freeze 하고,
      GAP feature 위에 작은 MLP + P-way sigmoid head를 올려 confidence만 학습.
    """
    gap_layer_name = f"resnet18_like_reg_{axis}_gap"
    try:
        gap_layer = base_model.get_layer(gap_layer_name)
    except ValueError as e:
        raise ValueError(f"GAP 레이어를 찾을 수 없습니다: {gap_layer_name}") from e

    # feature extractor (freeze)
    feature_model = keras.Model(base_model.input, gap_layer.output, name=f"{axis}_feature_model")
    feature_model.trainable = False

    # conf head
    inp = keras.Input(shape=input_shape, name=f"{axis}_input_for_conf")
    feat = feature_model(inp)
    x = layers.Dense(128, activation="relu", name=f"{axis}_conf_fc1")(feat)
    x = layers.Dense(64, activation="relu", name=f"{axis}_conf_fc2")(x)
    conf_out = layers.Dense(P, activation="sigmoid", name=f"{axis}_conf_out")(x)

    conf_model = keras.Model(inp, conf_out, name=f"{axis}_conf_model")
    # 타깃: 각 박스별 max IoU (연속값 0~1)를 회귀 형태로 근사
    conf_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return conf_model


def compute_conf_targets_for_axis(model: keras.Model, X, y, axis: str, roi_idx: int, K: int):
    """
    - 고정된 bbox 모델 `model` 로 prediction
    - 각 샘플/각 pred(p)에 대해: GT들과의 IoU 중 최대값을 confidence 타깃으로 사용
      ⇒ target_conf: (N, P) in [0, 1] (연속값, max IoU 자체를 예측하도록 학습)
    """
    _, y_reg = slice_roi_targets(y, roi_idx=roi_idx, K=K)
    gt_bbox = y_reg[:, : 4 * K].reshape(-1, K, 4)
    gt_mask = y_reg[:, 4 * K : 5 * K].reshape(-1, K)  # (N,K), 1이면 유효 GT

    pred = model.predict(X, batch_size=BATCH, verbose=1)
    pred_full = pred.reshape(-1, P, 5)  # [hc,hw,dc,dw,conf(orig)]
    pred_bbox = pred_full[:, :, :4]

    iou = iou_matrix_np(pred_bbox, gt_bbox)  # (N,P,K)
    # GT 없는 영역은 IoU=0으로 처리
    iou = np.where(gt_mask[:, None, :] > 0.5, iou, 0.0)

    # 각 샘플/각 pred에 대해 가능한 GT 중 최대 IoU 사용
    raw_iou_max = iou.max(axis=2)  # (N,P), 0~1

    # ------------------------------------------------------------------
    # 축별 IoU threshold를 이용해 "고품질 박스"를 더 강조하는 스케일링
    #   - IOU_POS_THRESHOLD_PER_AXIS[axis] = t 라고 할 때,
    #     t 이하는 0 근처로, t 이상~1.0 구간을 [0,1]로 선형 매핑
    #   - 예) t=0.4 면 IoU 0.4 → 0, IoU 1.0 → 1.0
    #   - 이렇게 하면 conf=1.0 이 "이 축에서 high-quality IoU" 에 더 직접적으로 대응됨.
    # ------------------------------------------------------------------
    thr = IOU_POS_THRESHOLD_PER_AXIS.get(axis, 0.5)
    scaled = (raw_iou_max - thr) / max(1e-6, (1.0 - thr))
    scaled = np.clip(scaled, 0.0, 1.0)

    # 연속 타깃: 축별 threshold를 반영한 max IoU 스케일 값 (0~1 실수)
    target_conf = scaled.astype("float32")
    return target_conf


def train_conf_for_axis(axis: str, roi_idx: int, X, y, K: int, conf_ckpt_dir: Path):
    print(f"\n===== Axis {axis.upper()} (ROI {roi_idx}) =====")

    best_path = ckpt_dir / f"best_{axis}.keras"
    if not best_path.exists():
        raise FileNotFoundError(f"best 모델 없음: {best_path}")

    print("Loading base model:", best_path)
    base_model = keras.models.load_model(str(best_path), compile=False)
    base_model.trainable = False  # 전체 freeze (안전장치)

    # -------------------------------
    # 1) break 샘플만 대상으로 사용
    # -------------------------------
    labels = y[:, 0].astype(int)
    mask_break = labels == 1
    if not np.any(mask_break):
        raise RuntimeError("break(label=1) 샘플이 없어 conf 학습을 진행할 수 없습니다.")

    X_break = X[mask_break]
    y_break = y[mask_break]
    print(f"break 샘플 수: {X_break.shape[0]} / 전체 {X.shape[0]}")

    # break 샘플에 대해 IoU 기반 confidence 타깃 계산
    print(
        "Computing IoU-based confidence targets "
        f"(axis={axis}, scaled by IOU_POS_THRESHOLD_PER_AXIS={IOU_POS_THRESHOLD_PER_AXIS.get(axis, 0.5)})..."
    )
    y_conf_full = compute_conf_targets_for_axis(
        base_model, X_break, y_break, axis=axis, roi_idx=roi_idx, K=K
    )
    print("y_conf_full shape:", y_conf_full.shape)  # (N_break,P)

    # train/val 인덱스 분할 (라벨 y_break[:,0] 기준 stratify)
    indices = np.arange(X_break.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=BASE_SEED,
        stratify=y_break[:, 0].astype(int),
    )

    X_train, X_val = X_break[train_idx], X_break[val_idx]
    yconf_train, yconf_val = y_conf_full[train_idx], y_conf_full[val_idx]

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    # conf 모델 구성 및 학습
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

    # conf 모델 저장 (2차 Hard run 디렉터리의 checkpoints 폴더에 저장)
    conf_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = conf_ckpt_dir / f"conf_{axis}.keras"
    conf_model.save(str(out_path))
    print(f"✅ Saved conf model for axis {axis} -> {out_path}")

    return history


# ============================================================================
# main
# ============================================================================

def main():
    global data_root, train_dir, test_dir, train_seq, train_lab, test_seq, test_lab

    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # 9. hard_train_data 최신 run 우선 사용, 없으면 5. train_data fallback
    hard_data_base = Path(current_dir) / "9. hard_train_data"
    data_run_dir = get_latest_hard_train_dir(hard_data_base)
    if data_run_dir is not None:
        data_root = data_run_dir
        train_dir = data_root / "train"
        test_dir = data_root / "test"
        train_seq = train_dir / "break_imgs_train.npy"
        train_lab = train_dir / "break_labels_train.npy"
        test_seq = test_dir / "break_imgs_test.npy"
        test_lab = test_dir / "break_labels_test.npy"
        print("학습 데이터(최신 hard run):", data_run_dir)
    else:
        data_root = DATA_FALLBACK_ROOT
        train_dir = data_root / "train"
        test_dir = data_root / "test"
        train_seq = train_dir / "break_imgs_train.npy"
        train_lab = train_dir / "break_labels_train.npy"
        test_seq = test_dir / "break_imgs_test.npy"
        test_lab = test_dir / "break_labels_test.npy"
        print("9. hard_train_data에 유효 run 없음 → 5. train_data 사용")

    if not train_seq.exists() or not train_lab.exists():
        raise FileNotFoundError(
            f"학습 데이터 없음: {train_seq}, {train_lab}\n"
            f"※ 먼저 9. set_hard_train_data.py 를 실행해 9. hard_train_data/<날짜>/train, test 에 NPY를 생성하세요."
        )

    X, y, X_test, y_test, K = load_data()
    print("X:", X.shape, "y:", y.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("K (#GT per ROI axis):", K)

    tf.keras.utils.set_random_seed(BASE_SEED)

    # 2차 Hard 모델(run) 디렉터리 생성: 12. hard_models_2nd/<timestamp>/
    second_stage_base = Path(current_dir) / "12. hard_models_2nd"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    second_stage_run_dir = second_stage_base / timestamp
    second_stage_run_dir.mkdir(parents=True, exist_ok=True)
    print("2차 Hard 모델 run 디렉터리:", second_stage_run_dir)

    # 2차 Hard run 전용 checkpoints 디렉터리 (conf_x/y/z.keras 저장 위치)
    second_ckpt_dir = second_stage_run_dir / "checkpoints"
    second_ckpt_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for axis, roi_idx in [("x", 0), ("y", 1), ("z", 2)]:
        hist = train_conf_for_axis(axis, roi_idx, X, y, K=K, conf_ckpt_dir=second_ckpt_dir)
        histories[axis] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    # 학습 설정/환경 저장 (2차 Hard 모델용)
    training_config = {
        "timestamp": timestamp,
        "second_stage_run": second_stage_run_dir.name,
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
            "train_val_split": {
                "test_size": 0.2,
                "random_state": BASE_SEED,
                "stratify": True,
            },
        },
        "model": {
            "architecture": "conf head on top of first-stage ResNet18-like bbox",
            "input_shape": list(X.shape[1:]),
            "pred_boxes_per_axis_P": P,
        },
        "loss": {
            "type": "mse (per-box conf ≈ IoU)",
            "iou_pos_threshold_per_axis": {k: float(v) for k, v in IOU_POS_THRESHOLD_PER_AXIS.items()},
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": 1e-3,
        },
        "first_stage_ckpt_dir": str(ckpt_dir),
    }
    with open(second_stage_run_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    with open(second_stage_run_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    print("2차 Hard 모델 학습 설정/히스토리 저장:", second_stage_run_dir)

    print("모든 축에 대해 conf head 학습 완료.")


if __name__ == "__main__":
    main()

