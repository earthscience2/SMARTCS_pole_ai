"""Light ResNet(2D) 파단 패턴 학습 - 전주 파단/정상 분류 모델 학습"""

import os
import sys
import subprocess
import datetime
import json
import argparse
from pathlib import Path

# 기본값(Windows): WSL2 실행. --local 이거나 이미 Linux/WSL 이면 이 프로세스에서 로컬 학습
_run_local = "--local" in sys.argv or sys.platform != "win32"
if _run_local:
    if "--local" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--local"]
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parent
    _sh_path = Path(_script_dir) / "7. make_light_model_wsl2.sh"
    if not _sh_path.exists():
        _sh_path = _project_root / "make_ai" / "7. make_light_model_wsl2.sh"
    if _sh_path.exists():
        _abs = _sh_path.resolve()
        _drive = _abs.drive
        _wsl_path = ("/mnt/" + _drive[0].lower() + str(_abs)[len(_drive):].replace("\\", "/")) if _drive else str(_abs).replace("\\", "/")
        print("WSL2에서 학습 실행:", _wsl_path)
        ret = subprocess.run(["wsl", "bash", _wsl_path] + sys.argv[1:], cwd=str(_project_root))
        sys.exit(ret.returncode)
    # 스크립트 없으면 로컬 학습 진행

import numpy as np
import matplotlib.pyplot as plt

# Windows에서 CUDA 라이브러리 경로 설정 (TensorFlow import 전에 실행)
if sys.platform == 'win32':
    # CUDA_PATH 환경 변수 확인
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        # 기본 CUDA 설치 경로 확인
        default_cuda_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0',
        ]
        for path in default_cuda_paths:
            if os.path.exists(path):
                cuda_path = path
                os.environ['CUDA_PATH'] = cuda_path
                break
    
    if cuda_path:
        # CUDA bin 경로를 PATH에 추가 (DLL 로딩을 위해)
        cuda_bin = os.path.join(cuda_path, 'bin')
        if os.path.exists(cuda_bin):
            current_path = os.environ.get('PATH', '')
            if cuda_bin not in current_path:
                os.environ['PATH'] = cuda_bin + os.pathsep + current_path
        
        # cuDNN 경로 확인 (일반적으로 CUDA 경로와 같거나 별도 설치)
        cudnn_bin = os.path.join(cuda_path, 'bin')
        if os.path.exists(cudnn_bin):
            current_path = os.environ.get('PATH', '')
            if cudnn_bin not in current_path:
                os.environ['PATH'] = cudnn_bin + os.pathsep + current_path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support
)

# 현재 스크립트 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="라이트 모델 학습")
parser.add_argument("--epochs", type=int, default=100, help="학습 epoch 수")
parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam 학습률")
parser.add_argument("--focal-alpha", type=float, default=0.93, help="Focal loss alpha 값")
parser.add_argument(
    "--break-class-weight-scale",
    type=float,
    default=1.35,
    help="파단 클래스 가중치 배율",
)
parser.add_argument("--run-tag", type=str, default="", help="run 폴더 이름 suffix")
args = parser.parse_args()

RUNTIME_EPOCHS = int(args.epochs)
RUNTIME_BATCH = int(args.batch_size)
RUNTIME_LR = float(args.learning_rate)
RUNTIME_FOCAL_ALPHA = float(args.focal_alpha)
RUNTIME_BREAK_WEIGHT_SCALE = float(args.break_class_weight_scale)
RUNTIME_RUN_TAG = str(args.run_tag).strip()


def get_latest_light_train_dir(base: Path):
    """base(6. light_train_data) 안에서 train/test NPY가 모두 있는 run 중 이름 기준 최신 폴더 반환.
    날짜 형식(YYYYMMDD_HHmm)이 아니어도 train+test NPY만 있으면 유효 run으로 인정."""
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


print("TF:", tf.__version__)

# ============================================================================
# GPU 설정
# ============================================================================

print(f"\n{'='*60}")
print("GPU 설정 및 진단")
print(f"{'='*60}")

# TensorFlow CUDA 지원 여부 확인
is_built_with_cuda = tf.test.is_built_with_cuda()
print(f"TensorFlow CUDA 빌드 여부: {is_built_with_cuda}")

# GPU 사용 가능 여부 확인
gpus = tf.config.list_physical_devices('GPU')
print(f"사용 가능한 GPU 개수: {len(gpus)}")

# 로그 디바이스 배치 확인
if len(gpus) == 0:
    print("\n⚠️  GPU를 찾을 수 없습니다.")
    print("\n진단 정보:")
    
    # 모든 물리 디바이스 확인
    all_devices = tf.config.list_physical_devices()
    print(f"  모든 디바이스: {all_devices}")
    
    # CUDA 라이브러리 경로 확인 (Windows)
    cuda_path = os.environ.get('CUDA_PATH', '설정되지 않음')
    print(f"  CUDA_PATH 환경 변수: {cuda_path}")
    
    # PATH에 CUDA가 포함되어 있는지 확인
    path_env = os.environ.get('PATH', '')
    cuda_in_path = 'CUDA' in path_env or 'cuda' in path_env
    print(f"  PATH에 CUDA 포함 여부: {cuda_in_path}")
    
    if not is_built_with_cuda:
        print("\n  ⚠️  TensorFlow가 CUDA 지원 없이 빌드되었습니다.")
        print("  해결 방법:")
        print("    1. tensorflow[and-cuda] 재설치: pip install tensorflow[and-cuda]")
        print("    2. 또는 CUDA/cuDNN을 수동으로 설치 후 TensorFlow 재설치")
        print("    3. WSL2에서 실행 → GPU 사용 가능. make_ai/WSL2_TensorFlow_GPU_설정_가이드.md 참고")
    else:
        print("\n  ⚠️  TensorFlow는 CUDA를 지원하지만 GPU를 찾지 못했습니다.")
        print("  가능한 원인:")
        print("    1. CUDA/cuDNN이 설치되지 않음")
        print("    2. CUDA 경로가 환경 변수에 설정되지 않음")
        print("    3. GPU 드라이버 문제")
        print("    4. 다른 프로세스가 GPU를 사용 중")
        print("  → WSL2에서 실행하면 GPU 사용이 쉬운 경우가 많습니다. make_ai/WSL2_TensorFlow_GPU_설정_가이드.md 참고")
    
    print("\n  CPU를 사용하여 학습합니다.")
else:
    print("GPU 디바이스:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    
    # GPU 메모리 증가 설정 (필요한 만큼만 동적으로 할당)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 메모리 증가 설정 완료")
    except RuntimeError as e:
        print(f"⚠️  GPU 메모리 설정 중 오류: {e}")
    
    # GPU를 사용하도록 명시적으로 설정
    print("✅ GPU를 사용하여 학습합니다.")

print(f"{'='*60}\n")


# ============================================================================
# 1) 데이터 로드 (6. light_train_data 중 가장 최근 날짜 폴더 사용)
# ============================================================================

# 스크립트 위치·프로젝트 루트·cwd 기준으로 6. light_train_data 후보 경로 순서 시도
_script_dir = Path(current_dir)
_light_data_candidates = [
    _script_dir / "6. light_train_data",
    _script_dir.parent / "6. light_train_data",
    Path.cwd() / "make_ai" / "6. light_train_data",
    Path.cwd() / "6. light_train_data",
]
light_data_base = None
data_run_dir = None
for cand in _light_data_candidates:
    if cand.exists():
        data_run_dir = get_latest_light_train_dir(cand)
        if data_run_dir is not None:
            light_data_base = cand
            break
if data_run_dir is None:
    # 원인 파악: 각 후보 경로별로 하위 폴더마다 train / test / metadata 유무 출력
    lines = []
    for cand in _light_data_candidates:
        exists = cand.exists()
        lines.append(f"  [{cand}]")
        lines.append(f"    exists={exists}")
        if exists:
            try:
                subs = sorted([d for d in cand.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                for d in subs:
                    ht = (d / "train" / "break_imgs_train.npy").exists()
                    ht2 = (d / "test" / "break_imgs_test.npy").exists()
                    meta = (d / "break_imgs_metadata.json").exists()
                    lines.append(f"    - {d.name}: train={ht}, test={ht2}, metadata={meta}")
            except Exception as e:
                lines.append(f"    iterdir 오류: {e}")
    hint = "\n".join(lines) if lines else "  (후보 경로 없음)"
    raise FileNotFoundError(
        f"학습용 NPY가 있는 run을 찾을 수 없습니다.\n"
        f"후보 경로별 상태 (current_dir={current_dir}, cwd={Path.cwd()}):\n{hint}\n\n"
        f"※ metadata=True 인데 train/test=False 이면, 해당 run에는 아직 NPY가 없습니다.\n"
        f"  → '6. set_light_train_data.py'를 끝까지 실행하면 train/, test/ NPY가 생성됩니다.\n"
        f"  → 이미 완료한 run이 있다면, 그 run 폴더에 train/·test/ 와 .npy 파일이 있어야 합니다."
    )
light_data_base = light_data_base or (data_run_dir.parent if data_run_dir else None)

train_dir = data_run_dir / "train"
test_dir = data_run_dir / "test"
train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"

print("사용 데이터(최신 run):", data_run_dir)
print("train_seq:", train_seq, "exists:", train_seq.exists())
print("train_lab:", train_lab, "exists:", train_lab.exists())
print("test_seq :", test_seq,  "exists:", test_seq.exists())
print("test_lab :", test_lab,  "exists:", test_lab.exists())

# 로드
X = np.load(train_seq).astype(np.float32)  # (N, 304, 19, 3)
y = np.load(train_lab).astype(np.float32)  # ✅ float32로 로드 (bh, bd 값 보존)

X_test = np.load(test_seq).astype(np.float32)
y_test = np.load(test_lab).astype(np.float32)  # ✅ float32로 로드 (bh, bd 값 보존)

print("X:", X.shape, X.dtype, "min/max:", float(X.min()), float(X.max()))
print("y:", y.shape, y.dtype, "counts:", np.unique(y[:, 0].astype(int), return_counts=True))
print("X_test:", X_test.shape, "y_test:", y_test.shape)

y_all_cls = y[:, 0].astype(int)
unique_labels, unique_counts = np.unique(y_all_cls, return_counts=True)
label_counts = {int(k): int(v) for k, v in zip(unique_labels, unique_counts)}

if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
    raise ValueError(
        "라이트 모델 학습 데이터에 단일 클래스만 있습니다. "
        f"현재 라벨 분포: {label_counts}. "
        f"데이터 run: {data_run_dir}. "
        "정상(0)과 파단(1)이 모두 포함되도록 "
        "'make_ai/6. set_light_train_data.py'를 다시 생성하세요."
    )

if label_counts.get(0, 0) < 2 or label_counts.get(1, 0) < 2:
    raise ValueError(
        "라이트 모델 학습 데이터의 클래스별 샘플 수가 너무 적습니다. "
        f"현재 라벨 분포: {label_counts}. "
        "각 클래스가 최소 2개 이상이어야 stratified split이 가능합니다."
    )


# ============================================================================
# 2) Train/Val split
# ============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[:, 0].astype(int)  # ✅ float32이므로 int로 변환하여 stratify
)

print("X_train:", X_train.shape, "y_train counts:", np.unique(y_train[:, 0].astype(int), return_counts=True))
print("X_val  :", X_val.shape,   "y_val counts  :", np.unique(y_val[:, 0].astype(int), return_counts=True))


# ============================================================================
# 3) 클래스 불균형 보정 (파단 FN 감소: 파단 클래스 가중치 추가 상향)
# ============================================================================

# 파단(1) 클래스 가중치 배율. 1.0=balanced만 사용, >1 로 올리면 파단 놓침(FN) 감소에 유리
# 1726(Recall 0.86, FN 7)에서 1.25 사용. 1.5는 1719에서 역효과 → 1.35로 소폭 상향해 FN 추가 감소 유도
BREAK_CLASS_WEIGHT_SCALE = RUNTIME_BREAK_WEIGHT_SCALE

classes = np.array([0, 1], dtype=np.int32)
y_train_cls = y_train[:, 0].astype(np.int32)   # (N,) 0/1만 추출
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls)

cw0 = float(cw[0])
cw1 = float(cw[1]) * BREAK_CLASS_WEIGHT_SCALE   # 파단 쪽 가중치 추가 강화

print(f"Class weights: 0={cw0:.4f}, 1={cw1:.4f} (break scale={BREAK_CLASS_WEIGHT_SCALE})")


# ============================================================================
# 4) 데이터셋 생성
# ============================================================================

BATCH = RUNTIME_BATCH
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(X, y, training: bool):
    """정상(0) / 파단(1) 라벨만 사용. 파단 위치(reg)는 학습하지 않음."""
    y_cls = y[:, 0:1].astype("float32")   # (N,1)

    # cls sample_weight: class imbalance 보정
    cls_int = y[:, 0].astype(np.int32)    # (N,)
    sw_cls = np.where(cls_int == 1, cw1, cw0).astype("float32")   # (N,)

    ds = tf.data.Dataset.from_tensor_slices((X, y_cls, sw_cls))
    if training:
        ds = ds.shuffle(min(len(X), 5000), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

ds_train = make_ds(X_train, y_train, training=True)
ds_val   = make_ds(X_val, y_val, training=False)
ds_test  = make_ds(X_test, y_test, training=False)


# ============================================================================
# 5) ResNet 모델 구축
# ============================================================================

def basic_block(x, filters, stride=(1,1)):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=(1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # projection shortcut
    if shortcut.shape[-1] != filters or stride != (1,1):
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def make_stage(x, filters, blocks, first_stride=(1,1)):
    x = basic_block(x, filters, stride=first_stride)
    for _ in range(blocks - 1):
        x = basic_block(x, filters, stride=(1,1))
    return x

def build_resnet18_like(input_shape=(304, 19, 3)):
    inp = keras.Input(shape=input_shape)

    # stem: width 보존 위해 stride (2,1)
    x = layers.Conv2D(64, 7, strides=(2,1), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=(2,1), padding="same")(x)

    # ResNet18 blocks: [2,2,2,2]
    x = make_stage(x, 64,  blocks=2, first_stride=(1,1))
    x = make_stage(x, 128, blocks=2, first_stride=(2,1))  # height down only
    x = make_stage(x, 256, blocks=2, first_stride=(2,1))
    x = make_stage(x, 512, blocks=2, first_stride=(2,1))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    cls_out = layers.Dense(1, activation="sigmoid", name="cls")(x)

    return keras.Model(inp, cls_out, name="resnet18_like")

model = build_resnet18_like(input_shape=X_train.shape[1:])
model.summary()


# ============================================================================
# 6) 모델 컴파일 (정상/파단 분류만) — alpha↑ 시 파단 놓침(FN) 감소에 유리
# ============================================================================

# alpha: 양성(파단) 클래스 가중치. 1726에서 0.92로 Recall 0.86. 0.95는 1719에서 역효과 → 0.93으로 소폭 상향
FOCAL_ALPHA = RUNTIME_FOCAL_ALPHA
FOCAL_GAMMA = 2.0

loss_cls = keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, from_logits=False)

model.compile(
    optimizer=keras.optimizers.Adam(RUNTIME_LR),
    loss=loss_cls,
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)


# ============================================================================
# 7) 콜백 설정
# ============================================================================

# 6. light_train_data와 동일한 날짜 형식(YYYYMMDD_HHmm)으로 결과 폴더 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
if RUNTIME_RUN_TAG:
    timestamp = f"{timestamp}_{RUNTIME_RUN_TAG}"
run_dir = Path(current_dir) / "7. light_models"
run_timestamp_dir = run_dir / timestamp

# 타임스탬프 폴더 내부 구조
ckpt_dir = run_timestamp_dir / "checkpoints"
log_dir = run_timestamp_dir / "logs"
results_dir = run_timestamp_dir / "results"

ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

best_ckpt_path = ckpt_dir / "best.keras"                # best 모델 저장(전체)
latest_weights = ckpt_dir / "latest.weights.h5"         # 매 epoch 최신 weights 저장

# val_loss 기준 저장·감속·조기종료 (권장)
# ※ val_recall 기준은 학습 초기(recall=1, 미수렴) 모델이 선택되어 threshold 붕괴·FN 역증가하는 문제 있음 → 사용 안 함
# FN 감소는 BREAK_CLASS_WEIGHT_SCALE, FOCAL_ALPHA 상향으로만 반영
MONITOR = "val_loss"
MODE = "min"

cb_best = keras.callbacks.ModelCheckpoint(
    filepath=str(best_ckpt_path),
    monitor=MONITOR,
    mode=MODE,
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


class SaveLatestWeights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(str(latest_weights))

cb_latest = SaveLatestWeights()

cb_tb = keras.callbacks.TensorBoard(log_dir=str(log_dir))

# patience 여유를 주어 1537처럼 26 epoch 근처까지 수렴할 기회 확보 (1719는 20 epoch에서 조기 종료됨)
cb_rlr = keras.callbacks.ReduceLROnPlateau(
    monitor=MONITOR, mode=MODE, factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

cb_es = keras.callbacks.EarlyStopping(
    monitor=MONITOR, mode=MODE, patience=12, restore_best_weights=True, verbose=1
)

callbacks = [cb_best, cb_latest, cb_tb, cb_rlr, cb_es]

print(f"✅ run_dir: {run_dir}")
print(f"✅ timestamp: {timestamp}")
print(f"✅ run_timestamp_dir: {run_timestamp_dir}")
print(f"✅ best_ckpt_path: {best_ckpt_path}")
print(f"✅ latest_weights: {latest_weights}")
print(f"✅ tensorboard log_dir: {log_dir}")
print(f"✅ results_dir: {results_dir}")
print(
    f"✅ runtime args: epochs={RUNTIME_EPOCHS}, batch={RUNTIME_BATCH}, "
    f"lr={RUNTIME_LR}, focal_alpha={RUNTIME_FOCAL_ALPHA}, "
    f"break_weight_scale={RUNTIME_BREAK_WEIGHT_SCALE}"
)


# ============================================================================
# 8) 학습 조건 저장
# ============================================================================

# 학습 하이퍼파라미터 정의
EPOCHS = RUNTIME_EPOCHS

# 학습 시작 전에 조건 저장
training_config = {
    "timestamp": timestamp,
    "training_start_time": datetime.datetime.now().isoformat(),
    "data_source_run": str(data_run_dir.name),
    "data_source_path": str(data_run_dir),
    "data": {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_break_count": int((y_train[:, 0] == 1).sum()),
        "train_normal_count": int((y_train[:, 0] == 0).sum()),
        "val_break_count": int((y_val[:, 0] == 1).sum()),
        "val_normal_count": int((y_val[:, 0] == 0).sum()),
        "test_break_count": int((y_test[:, 0] == 1).sum()),
        "test_normal_count": int((y_test[:, 0] == 0).sum()),
        "input_shape": list(X_train.shape[1:]),
    },
    "model": {
        "architecture": "ResNet18-like",
        "input_shape": list(X_train.shape[1:]),
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "train_val_split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
    },
    "class_weights": {
        "normal": float(cw0),
        "break": float(cw1),
        "break_class_weight_scale": BREAK_CLASS_WEIGHT_SCALE,
    },
    "loss": {
        "type": "BinaryFocalCrossentropy",
        "gamma": FOCAL_GAMMA,
        "alpha": FOCAL_ALPHA,
        "from_logits": False,
    },
    "optimizer": {
        "type": "Adam",
        "learning_rate": RUNTIME_LR,
    },
    "callbacks": {
        "monitor": MONITOR,
        "mode": MODE,
        "model_checkpoint": {
            "save_best_only": True,
            "save_weights_only": False,
        },
        "reduce_lr": {
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
        "early_stopping": {
            "patience": 12,
            "restore_best_weights": True,
        },
    },
    "metrics": ["acc", "precision", "recall", "auc", "pr_auc"],
}

config_path = results_dir / "training_config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)
print(f"\n✅ 학습 조건 저장: {config_path}")


# ============================================================================
# 9) 학습
# ============================================================================

print(f"\n{'='*60}")
print("학습 시작")
print(f"{'='*60}")

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# 학습 완료 시간 기록
training_config["training_end_time"] = datetime.datetime.now().isoformat()
training_config["total_epochs_trained"] = len(history.history.get("loss", []))
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

# 최고 모델 로드 (분류 전용 단일 출력 모델)
print(f"Loading best model: {best_ckpt_path}")
try:
    best_model = keras.models.load_model(str(best_ckpt_path), compile=False)
except Exception as e:
    print(f"로드 실패: {e}. 구조를 다시 빌드하고 weights만 로드합니다...")
    best_model = build_resnet18_like(input_shape=X_train.shape[1:])
    best_model.load_weights(str(best_ckpt_path).replace(".keras", ".weights.h5").replace("best", "latest"))
    print("⚠️  최신 weights를 로드했습니다. best weights가 아닐 수 있습니다.")

# 학습 히스토리 저장
history_dict = {}
for key, values in history.history.items():
    history_dict[key] = [float(v) for v in values]

history_path = results_dir / "training_history.json"
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history_dict, f, ensure_ascii=False, indent=2)
print(f"\n✅ 학습 히스토리 저장: {history_path}")


# ============================================================================
# 10) 평가 - Classification
# ============================================================================

# val에서 최적 threshold 찾기 (단일 출력 모델: predict는 (N,1) 반환)
pred_val = best_model.predict(X_val, batch_size=BATCH)
y_prob_val = np.asarray(pred_val).reshape(-1)

y_val_cls = y_val[:, 0].astype(int)

P_MIN = 0.50
ths = np.linspace(0.01, 0.99, 199)

best = None
for th in ths:
    y_pred_val = (y_prob_val >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val_cls, y_pred_val, average="binary", pos_label=1, zero_division=0
    )
    if p >= P_MIN and (best is None or r > best[2]):
        best = (th, p, r, f1)

print("best(th, p, r, f1) =", best)
best_th = best[0] if best else 0.5

# test에서 평가
pred_test = best_model.predict(X_test, batch_size=BATCH)
y_prob_test = np.asarray(pred_test).reshape(-1)
y_pred_test = (y_prob_test >= best_th).astype(int)

y_test_cls = y_test[:, 0].astype(int)

# 지표 출력
p = precision_score(y_test_cls, y_pred_test, zero_division=0)
r = recall_score(y_test_cls, y_pred_test, zero_division=0)
f1 = f1_score(y_test_cls, y_pred_test, zero_division=0)
auc = roc_auc_score(y_test_cls, y_prob_test)

print(f"\nbest_th = {best_th:.4f}")
print(f"Precision = {p:.4f} | Recall = {r:.4f} | F1 = {f1:.4f} | ROC-AUC = {auc:.4f}")
print("\nClassification report:\n", classification_report(y_test_cls, y_pred_test, digits=4, zero_division=0))

# 혼동행렬
cm = confusion_matrix(y_test_cls, y_pred_test)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

# 평가 결과 저장 (분류만)
evaluation_results = {
    "timestamp": timestamp,
    "evaluation_time": datetime.datetime.now().isoformat(),
    "model_path": str(best_ckpt_path),
    "test_samples": int(len(X_test)),
    "classification": {
        "threshold": float(best_th),
        "precision": float(p),
        "recall": float(r),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    },
}

plt.figure(figsize=(5,4))
plt.imshow(cm_norm)
plt.title("Confusion Matrix (row-normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1], ["0", "1"])
plt.yticks([0,1], ["0", "1"])

# 숫자(개수) + 퍼센트 같이 표기
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        plt.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center")

plt.tight_layout()
confusion_matrix_path = results_dir / "confusion_matrix.png"
plt.savefig(confusion_matrix_path)
print(f"\nConfusion matrix saved to: {confusion_matrix_path}")

eval_results_path = results_dir / "evaluation_results.json"
with open(eval_results_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
print(f"\n✅ 평가 결과 저장: {eval_results_path}")

print("\n학습 완료!")
print(f"최고 모델 저장 위치: {best_ckpt_path}")
print(f"결과 저장 위치: {results_dir}")
print(f"타임스탬프: {timestamp}")
