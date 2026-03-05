"""Light ResNet(2D) 모델 학습 및 평가 + 베스트 모델 선택 (결과는 2. light_models/<run>/evaluation/ 폴더에 저장됨)

실행 방법:
- 기본: python 2. make_light_model.py
  -> 먼저 WSL2에서 GPU 스크립트 시도, 기본은 여기서 종료(재실행 없음)
     (필요 시 LIGHT_WSL_FALLBACK_CPU=1 로 CPU fallback 허용)
- CPU 강제: python 2. make_light_model.py --cpu
  -> GPU 스크립트 건너뛰고 바로 CPU로 실행
- 로컬 모드: python 2. make_light_model.py --local
  -> 자동 재학습 없이 한 번만 실행 (CPU)
- 자동 재학습: python 2. make_light_model.py --auto-retrain
  -> 평가 기반 자동 재학습 활성화
"""

import os
import sys
import subprocess
import datetime
import json
import argparse
import importlib.util
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger, log_event

LOGGER = get_logger("train_light_model")


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
    allow_cpu_fallback = os.environ.get("LIGHT_WSL_FALLBACK_CPU", "0") == "1"
        
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = Path(_script_dir).parents[1]  # 2단계 위로 올라가서 프로젝트 루트
    _sh_path = Path(_script_dir) / "2. make_light_model_gpu.sh"
    
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
        
        # 현재 스크립트의 모든 인수를 전달 (--local, --auto-retrain 제외)
        filtered_args = [a for a in sys.argv[1:] if a not in ["--local", "--auto-retrain"]]
        
        ret = subprocess.run(["wsl", "bash", _wsl_path] + filtered_args, 
                           cwd=str(_project_root))
        
        if ret.returncode == 0:
            print("=" * 60)
            print("WSL2 GPU 스크립트가 성공적으로 완료되었습니다!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("=" * 60)
            print(f"WSL2 GPU 스크립트가 실패했습니다 (exit code: {ret.returncode})")
            if allow_cpu_fallback:
                print("CPU 모드로 fallback합니다... (LIGHT_WSL_FALLBACK_CPU=1)")
                print("=" * 60)
                return False
            print("같은 작업을 CPU로 재실행하지 않습니다(기본). 필요 시 --cpu 로 CPU만 실행하거나, LIGHT_WSL_FALLBACK_CPU=1 로 fallback을 켜세요.")
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

# 기본적으로 자동 재학습 비활성화 (한 번만 학습)
# 자동 재학습을 활성화하려면 --auto-retrain 옵션을 사용하세요
_enable_auto_retrain = "--auto-retrain" in sys.argv
_run_local = "--local" in sys.argv
_force_cpu = "--cpu" in sys.argv

# 명령행 인수 정리
if "--local" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--local"]
if "--auto-retrain" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--auto-retrain"]
if "--cpu" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--cpu"]

# GPU 스크립트 시도 (--local 또는 --cpu 옵션이 없는 경우)
if not _run_local and not _force_cpu:
    if try_wsl2_gpu_script():
        # GPU 스크립트가 성공하면 여기서 종료됨
        pass

# CPU 모드로 계속 진행
if _run_local:
    print("로컬 모드: 한 번만 학습합니다 (자동 재학습 비활성화)")
elif _force_cpu:
    print("CPU 강제 모드: GPU 스크립트를 건너뛰고 CPU로 실행합니다")
else:
    print("CPU 모드로 fallback하여 실행합니다") 

import numpy as np
import matplotlib.pyplot as plt

# Windows에서 CUDA 환경 설정 (TensorFlow import 전에 실행)
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
        # CUDA bin 디렉토리를 PATH에 추가 (DLL 로딩을 위해)
        cuda_bin = os.path.join(cuda_path, 'bin')
        if os.path.exists(cuda_bin):
            current_path = os.environ.get('PATH', '')
            if cuda_bin not in current_path:
                os.environ['PATH'] = cuda_bin + os.pathsep + current_path
        
        # cuDNN 라이브러리 경로 (일반적으로 CUDA 설치 경로와 동일)
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
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)
import sklearn.metrics as sk_metrics

# 데이터셋 디렉토리 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_SET_DIR = Path(current_dir).parent / "1. make_data set"

# =========================
# USER OPTIONS (edit here)
# =========================
USER_OPTIONS = {
    # [주요 파라미터] 중복 검사(signature)에 포함됨
    # - data_source_run(데이터 run 폴더명)은 자동 포함(여기에는 없음)
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "focal_alpha": 0.96,
    "break_class_weight_scale": 1.4,

    # [서브 파라미터] 중복 검사(signature)에는 미포함 (평가/선정 기준 및 실행 옵션)
    "target_precision": 0.81,
    "target_recall": 0.90,
    "target_f1": 0.84,
    "target_pass_mode": "all_metrics",   # all_metrics | recall_priority
    "best_target_recall": 0.90,
    "best_target_accuracy": 0.50,
    "skip_evaluation": False,
    # 고정 분류 임계값 사용 시 threshold 값
    "fixed_threshold": 0.50,
}

parser = argparse.ArgumentParser(description="경량 모델 학습 + 베스트 모델 선택")
parser.add_argument("--epochs", type=int, default=USER_OPTIONS["epochs"], help="학습 epoch 수")
parser.add_argument("--batch-size", type=int, default=USER_OPTIONS["batch_size"], help="배치 크기")
parser.add_argument("--learning-rate", type=float, default=USER_OPTIONS["learning_rate"], help="Adam 학습률")
parser.add_argument("--focal-alpha", type=float, default=USER_OPTIONS["focal_alpha"], help="Focal loss alpha 값")
parser.add_argument(
    "--break-class-weight-scale",
    type=float,
    default=USER_OPTIONS["break_class_weight_scale"],
    help="파단 클래스 가중치 스케일",
)
# 목표 성능 지표 (베스트 모델 선택 기준)
parser.add_argument("--target-precision", type=float, default=USER_OPTIONS["target_precision"], help="목표 precision")
parser.add_argument("--target-recall", type=float, default=USER_OPTIONS["target_recall"], help="목표 recall")
parser.add_argument("--target-f1", type=float, default=USER_OPTIONS["target_f1"], help="목표 F1")
parser.add_argument("--target-pass-mode", type=str, choices=["all_metrics", "recall_priority"], default=USER_OPTIONS["target_pass_mode"])
parser.add_argument("--best-target-recall", type=float, default=USER_OPTIONS["best_target_recall"])
parser.add_argument("--best-target-accuracy", type=float, default=USER_OPTIONS["best_target_accuracy"])
parser.add_argument("--skip-evaluation", action="store_true", default=USER_OPTIONS["skip_evaluation"], help="평가 단계 건너뛰기")
parser.add_argument("--fixed-threshold", type=float, default=USER_OPTIONS["fixed_threshold"], help="고정 분류 임계값(threshold)")
parser.add_argument("--auto-retrain", action="store_true", help="자동 재학습 활성화 (성능 미달 시 파라미터 조정하여 재학습)")
args = parser.parse_args()

RUNTIME_EPOCHS = int(args.epochs)
RUNTIME_BATCH = int(args.batch_size)
RUNTIME_LR = float(args.learning_rate)
RUNTIME_FOCAL_ALPHA = float(args.focal_alpha)
RUNTIME_BREAK_WEIGHT_SCALE = float(args.break_class_weight_scale)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(v):
    return round(float(v), 10)


def _build_light_signature(data_run_name: str):
    return {
        "data_source_run": str(data_run_name),
        "epochs": int(RUNTIME_EPOCHS),
        "batch_size": int(RUNTIME_BATCH),
        "learning_rate": _round_float(RUNTIME_LR),
        "focal_alpha": _round_float(RUNTIME_FOCAL_ALPHA),
        "break_class_weight_scale": _round_float(RUNTIME_BREAK_WEIGHT_SCALE),
    }


def _extract_light_signature_from_config(cfg: dict):
    try:
        return {
            "data_source_run": str(cfg.get("data_source_run")),
            "epochs": int(cfg.get("training", {}).get("epochs")),
            "batch_size": int(cfg.get("training", {}).get("batch_size")),
            "learning_rate": _round_float(cfg.get("optimizer", {}).get("learning_rate")),
            "focal_alpha": _round_float(cfg.get("loss", {}).get("alpha")),
            "break_class_weight_scale": _round_float(cfg.get("class_weights", {}).get("break_class_weight_scale")),
        }
    except Exception:
        return None


def _find_duplicate_light_run(models_base: Path, signature: dict):
    if not models_base.exists():
        return None
    for run_dir in sorted([d for d in models_base.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True):
        cfg_path = run_dir / "results" / "training_config.json"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        old_sig = _extract_light_signature_from_config(cfg)
        if old_sig is not None and old_sig == signature:
            return run_dir.name
    return None


def _rank_key(metrics: dict, target_recall: float, target_accuracy: float):
    recall = _to_float(metrics.get("recall"))
    accuracy = _to_float(metrics.get("accuracy"))
    f1 = _to_float(metrics.get("f1"))
    meets = int((recall is not None and accuracy is not None and recall >= target_recall and accuracy >= target_accuracy))
    return (meets, (recall - target_recall) if recall is not None else -1e9, (accuracy - target_accuracy) if accuracy is not None else -1e9, recall or -1e9, accuracy or -1e9, f1 or -1e9)


def _load_set_light_train_data_module():
    script_path = Path(current_dir) / "1. set_light_train_data.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("set_light_train_data", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["set_light_train_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_test_data_from_merge_and_edit(
    merge_data_dir="4. merge_data",
    edit_data_dir="5. edit_data",
    min_points=200,
    max_points=400,
    sort_by="height",
    max_normal_samples=5000,
):
    std = _load_set_light_train_data_module()
    if std is None:
        return None, None, None
    from tqdm import tqdm
    prepare_sequence_from_csv = std.prepare_sequence_from_csv
    resize_img_height = std.resize_img_height
    crop_files = std.collect_break_files_from_edit_data(edit_data_dir=edit_data_dir, merge_data_dir=merge_data_dir)
    normal_files = std.collect_all_crop_files(merge_data_dir, is_break=False)
    imgs, labels_cls, csv_paths = [], [], []
    target_h = 304
    for csv_path, _pn, _pid, label in tqdm(crop_files, desc="  [break] CSV 전처리", unit="file"):
        result = prepare_sequence_from_csv(csv_path=csv_path, sort_by=sort_by, feature_min_max=None)
        if result is None:
            continue
        img, meta = result
        if meta.get("original_length", 0) < min_points or meta.get("original_length", 0) > max_points:
            continue
        imgs.append(resize_img_height(img, target_h=target_h))
        labels_cls.append(label)
        csv_paths.append(str(csv_path))
    n_kept = 0
    for csv_path, _pn, _pid, label in tqdm(normal_files, desc="  [normal] CSV 전처리", unit="file"):
        if n_kept >= max_normal_samples:
            break
        result = prepare_sequence_from_csv(csv_path=csv_path, sort_by=sort_by, feature_min_max=None)
        if result is None:
            continue
        img, meta = result
        if meta.get("original_length", 0) < min_points or meta.get("original_length", 0) > max_points:
            continue
        imgs.append(resize_img_height(img, target_h=target_h))
        labels_cls.append(label)
        csv_paths.append(str(csv_path))
        n_kept += 1
    if not imgs:
        return None, None, None
    return np.array(imgs, dtype=np.float32), np.array(labels_cls, dtype=np.int32), csv_paths


def _extract_candidate_from_report(report_path: Path):
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    model_run = data.get("model_run")
    cls = data.get("classification", {})
    metrics = {"accuracy": _to_float(cls.get("accuracy")), "precision": _to_float(cls.get("precision")), "recall": _to_float(cls.get("recall")), "f1": _to_float(cls.get("f1_score"))}
    if not model_run or metrics["recall"] is None or metrics["f1"] is None or metrics["accuracy"] is None:
        return None
    return {"model_run": model_run, "metrics": metrics, "report_path": str(report_path), "evaluation_time": data.get("evaluation_time")}


def _collect_best_candidate_from_reports(eval_base_dir: Path, target_recall: float, target_accuracy: float):
    report_paths = list(eval_base_dir.glob("*/evaluation_report.json")) + list(eval_base_dir.glob("*/evaluation/evaluation_report.json"))
    report_paths = [p for p in report_paths if p.exists()]
    if not report_paths:
        return None
    best_by_run = {}
    for report_path in report_paths:
        c = _extract_candidate_from_report(report_path)
        if c is None:
            continue
        r = c["model_run"]
        if r not in best_by_run or _rank_key(c["metrics"], target_recall, target_accuracy) > _rank_key(best_by_run[r]["metrics"], target_recall, target_accuracy):
            best_by_run[r] = c
    if not best_by_run:
        return None
    return max(best_by_run.values(), key=lambda c: _rank_key(c["metrics"], target_recall, target_accuracy))


def _load_current_best_metrics(best_alias_dir: Path):
    meta_path = best_alias_dir / "best_model_selection.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected = meta.get("selected", {})
            m = selected.get("metrics", {})
            return {"model_run": selected.get("model_run"), "metrics": {k: _to_float(m.get(k)) for k in ["accuracy", "precision", "recall", "f1"]}}
        except Exception:
            pass
    eval_result_path = best_alias_dir / "results" / "evaluation_results.json"
    if not eval_result_path.exists():
        # Support nested best alias layout: best_light_model/<run>/results/evaluation_results.json
        nested_reports = list(best_alias_dir.glob("*/results/evaluation_results.json"))
        if not nested_reports:
            return None
        eval_result_path = max(nested_reports, key=lambda p: p.parent.parent.name)
    try:
        with open(eval_result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls = data.get("classification", {})
        acc = _to_float(cls.get("accuracy"))
        cm = cls.get("confusion_matrix")
        if acc is None and isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
            total = float(np.sum(cm))
            if total > 0:
                acc = (float(cm[0][0]) + float(cm[1][1])) / total
        return {"model_run": data.get("model_run"), "metrics": {"accuracy": acc, "precision": _to_float(cls.get("precision")), "recall": _to_float(cls.get("recall")), "f1": _to_float(cls.get("f1_score"))}}
    except Exception:
        return None



def _append_best_selection_history(models_base: Path, record: dict):
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
def get_latest_light_train_dir(base: Path):
    """base(1. light_train_data) 폴더에서 train/test NPY 파일이 있는 run 중 가장 최신 것을 반환.
    런 이름(YYYYMMDD_HHmm) 기준으로 정렬하여 train+test NPY가 모두 있는 run을 선택."""
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


def load_all_data_from_light_train_run(run_dir: Path):
    """1. light_train_data/<run> 에서 전처리된 전체 데이터(NPY)를 로드."""
    if run_dir is None:
        return None, None
    data_seq = run_dir / "break_imgs.npy"
    data_lab = run_dir / "break_labels.npy"
    if not data_seq.exists() or not data_lab.exists():
        return None, None
    X_all = np.load(data_seq).astype(np.float32)
    y_all = np.load(data_lab).astype(np.float32)
    return X_all, y_all


print("Python:", sys.executable)
print("TF:", tf.__version__)

# ============================================================================
# GPU 설정
# ============================================================================

print(f"\n{'='*60}")
print("GPU 환경 확인")
print(f"{'='*60}")

# TensorFlow CUDA 지원 여부 확인
is_built_with_cuda = tf.test.is_built_with_cuda()
print(f"TensorFlow CUDA 지원 여부: {is_built_with_cuda}")

# GPU 디바이스 개수 확인
gpus = tf.config.list_physical_devices('GPU')
print(f"사용 가능한 GPU 개수: {len(gpus)}")

# GPU가 없는 경우 처리
if len(gpus) == 0:
    print("\n사용 가능한 GPU가 없습니다.")
    print("\n진단 정보:")
    
    # 모든 디바이스 확인
    all_devices = tf.config.list_physical_devices()
    print(f"  모든 디바이스: {all_devices}")
    
    # CUDA 환경 변수 확인 (Windows)
    cuda_path = os.environ.get('CUDA_PATH', '설정되지 않음')
    print(f"  CUDA_PATH 값: {cuda_path}")
    
    # PATH에서 CUDA 관련 경로 확인
    path_env = os.environ.get('PATH', '')
    cuda_in_path = 'CUDA' in path_env or 'cuda' in path_env
    print(f"  PATH에서 CUDA 경로 발견: {cuda_in_path}")
    
    if not is_built_with_cuda:
        print("\n  문제: TensorFlow가 CUDA 지원 없이 설치됨")
        print("  해결 방법:")
        if sys.platform == "win32":
            print("    - Windows용 TensorFlow(pip: tensorflow)는 현재 환경에서 GPU(CUDA)를 지원하지 않습니다.")
            print("    - 권장: WSL2에서 실행해 GPU를 사용하세요 (main/2. make_light_model/2. make_light_model_gpu.sh).")
            print("    - 대안: Linux 머신/서버에서 학습 또는 다른 프레임워크(PyTorch CUDA) 사용.")
        else:
            print("    1. tensorflow[and-cuda] 설치: pip install 'tensorflow[and-cuda]'")
            print("    2. 또는 CUDA/cuDNN을 수동으로 설치 후 TensorFlow 재설치")
            print("    3. WSL2/리눅스 환경에서 실행해 GPU를 사용하는 방법을 확인하세요: main/2. make_light_model/")
    else:
        print("\n  문제: TensorFlow는 CUDA를 지원하지만 GPU를 인식하지 못함")
        print("  가능한 원인:")
        print("    1. CUDA/cuDNN 버전 불일치")
        print("    2. CUDA 드라이버 또는 런타임 문제")
        print("    3. GPU 드라이버 문제")
        print("    4. 다른 프로세스가 GPU를 사용 중")
        print("  WSL2에서 실행하면 GPU 인식률이 더 높습니다. main/2. make_light_model/ 스크립트를 참고하세요.")
    
    print("\n  CPU 모드로 계속 진행합니다.")
else:
    print("GPU 목록:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    
    # GPU 메모리 증가 허용 (메모리 부족 방지)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 메모리 증가 허용 설정 완료")
    except RuntimeError as e:
        print(f"GPU 메모리 설정 실패: {e}")
    
    # GPU 모드로 학습 진행
    print("GPU 모드로 학습을 시작합니다.")

print(f"{'='*60}\n")


# ============================================================================
# 1) 데이터 로딩 (1. light_train_data 폴더에서 최신 런 찾기)
# ============================================================================

# 여러 후보 위치에서 cwd 기준으로 1. light_train_data 폴더 찾기
_script_dir = Path(current_dir)
_light_data_candidates = [
    _script_dir / "1. light_train_data",
    _script_dir.parent / "2. make_light_model" / "1. light_train_data",
    Path.cwd() / "main" / "2. make_light_model" / "1. light_train_data",
    Path.cwd() / "1. light_train_data",
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
    # 디버그 정보: 각 후보 경로의 train / test / metadata 파일 상태 
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
    hint = "\n".join(lines) if lines else "  (정보 없음)"
    raise FileNotFoundError(
        f"학습용 NPY 파일이 있는 run을 찾을 수 없습니다.\n"
        f"검색 결과 (current_dir={current_dir}, cwd={Path.cwd()}):\n{hint}\n\n"
        f"해결방법: metadata=True로 train/test=False 옵션으로, 먼저 run을 생성한 후 NPY 파일을 생성하세요.\n"
        f"  1단계: '1. set_light_train_data.py'를 실행해 train/, test/ NPY 파일 생성\n"
        f"  2단계: 생성된 타임스탬프 run 폴더에서 해당 run 내의 train/test/ 하위 .npy 파일들이 모두 존재하는지 확인"
    )
light_data_base = light_data_base or (data_run_dir.parent if data_run_dir else None)

train_dir = data_run_dir / "train"
test_dir = data_run_dir / "test"
train_seq = train_dir / "break_imgs_train.npy"
train_lab = train_dir / "break_labels_train.npy"
test_seq = test_dir / "break_imgs_test.npy"
test_lab = test_dir / "break_labels_test.npy"

print("선택된 데이터 run:", data_run_dir)
print("train_seq:", train_seq, "exists:", train_seq.exists())
print("train_lab:", train_lab, "exists:", train_lab.exists())
print("test_seq :", test_seq,  "exists:", test_seq.exists())
print("test_lab :", test_lab,  "exists:", test_lab.exists())

light_models_base = Path(current_dir) / "2. light_models"
light_signature = _build_light_signature(str(data_run_dir.name))
duplicate_light_run = _find_duplicate_light_run(light_models_base, light_signature)
if duplicate_light_run is not None:
    print(
        f"중복 파라미터 감지: 기존 run '{duplicate_light_run}'와 동일한 학습 설정입니다. "
        "이번 실행은 학습/평가를 건너뜁니다."
    )
    sys.exit(0)


# 로드
X = np.load(train_seq).astype(np.float32)  # (N, 304, 19, 3)
y = np.load(train_lab).astype(np.float32)  # (N, M) float32 (0번째 컬럼: cls 0/1)

X_test = np.load(test_seq).astype(np.float32)
y_test = np.load(test_lab).astype(np.float32)  # (N, M) float32 (0번째 컬럼: cls 0/1)

print("X:", X.shape, X.dtype, "min/max:", float(X.min()), float(X.max()))
print("y:", y.shape, y.dtype, "counts:", np.unique(y[:, 0].astype(int), return_counts=True))
print("X_test:", X_test.shape, "y_test:", y_test.shape)

y_all_cls = y[:, 0].astype(int)
unique_labels, unique_counts = np.unique(y_all_cls, return_counts=True)
label_counts = {int(k): int(v) for k, v in zip(unique_labels, unique_counts)}

if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
    raise ValueError(
        "학습 데이터에 정상(0) 또는 파단(1) 클래스가 없습니다. "
        f"label counts: {label_counts}. "
        f"data run: {data_run_dir}. "
        "'1. set_light_train_data.py'를 다시 실행해 데이터를 생성하세요."
    )

if label_counts.get(0, 0) < 2 or label_counts.get(1, 0) < 2:
    raise ValueError(
        "클래스별 샘플 수가 너무 적습니다(클래스당 최소 2개 필요). "
        f"label counts: {label_counts}. "
        "stratified split을 할 수 없으니 데이터를 추가하거나 split 비율을 조정하세요."
    )


# ============================================================================
# 2) Train/Val split
# ============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[:, 0].astype(int)  # y는 float32이지만 stratify는 int로
)

print("X_train:", X_train.shape, "y_train counts:", np.unique(y_train[:, 0].astype(int), return_counts=True))
print("X_val  :", X_val.shape,   "y_val counts  :", np.unique(y_val[:, 0].astype(int), return_counts=True))


# ============================================================================
# 3) Class weight (FN 감소 목적)
# ============================================================================

# class 1(파단)에 가중치 스케일을 곱해 FN(미검출) 감소를 유도
# scale을 키울수록 recall은 올라가고 precision은 내려갈 수 있음
BREAK_CLASS_WEIGHT_SCALE = RUNTIME_BREAK_WEIGHT_SCALE

classes = np.array([0, 1], dtype=np.int32)
y_train_cls = y_train[:, 0].astype(np.int32)   # (N,) 0/1?
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_cls)

cw0 = float(cw[0])
cw1 = float(cw[1]) * BREAK_CLASS_WEIGHT_SCALE   # ? ? ? 

print(f"Class weights: 0={cw0:.4f}, 1={cw1:.4f} (break scale={BREAK_CLASS_WEIGHT_SCALE})")


# ============================================================================
# 4) tf.data 파이프라인
# ============================================================================

BATCH = RUNTIME_BATCH
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(X, y, training: bool):
    """정상(0) / 파단(1) 분류용 데이터셋을 만듭니다."""
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
# 5) ResNet 모델 구성
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

    # stem: width  ? stride (2,1)
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
# 6) Loss 설정 (Focal loss; FN 감소 유도)
# ============================================================================

# alpha: 파단(1) 쪽 가중치. 값이 클수록 FN 감소(=recall 증가) 경향
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
# 7) 콜백/출력 경로
# ============================================================================

# 실행 시각(YYYYMMDD_HHmm) 기준 run 폴더 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
run_dir = Path(current_dir) / "2. light_models"
run_timestamp_dir = run_dir / timestamp

# 출력 디렉토리
ckpt_dir = run_timestamp_dir / "checkpoints"
log_dir = run_timestamp_dir / "logs"
results_dir = run_timestamp_dir / "results"

# 디렉토리 생성 및 확인
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# 디렉토리 생성 확인
if not ckpt_dir.exists():
    raise RuntimeError(f"체크포인트 디렉토리 생성 실패: {ckpt_dir}")
if not log_dir.exists():
    raise RuntimeError(f"로그 디렉토리 생성 실패: {log_dir}")
if not results_dir.exists():
    raise RuntimeError(f"결과 디렉토리 생성 실패: {results_dir}")

print(f"디렉토리 생성 완료:")
print(f"  - 체크포인트: {ckpt_dir}")
print(f"  - 로그: {log_dir}")
print(f"  - 결과: {results_dir}")

best_ckpt_path = ckpt_dir / "best.keras"                # best 모델(전체) 저장
latest_weights = ckpt_dir / "latest.weights.h5"         # 매 epoch weights 저장
# best 선택 기준
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

# TensorBoard 콜백 생성 (디렉토리 존재 확인 후)
if not log_dir.exists() or not log_dir.is_dir():
    raise RuntimeError(f"TensorBoard 로그 디렉토리가 존재하지 않거나 디렉토리가 아닙니다: {log_dir}")

cb_tb = keras.callbacks.TensorBoard(
    log_dir=str(log_dir),
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq='epoch'
)

# ReduceLROnPlateau/early stopping 설정
cb_rlr = keras.callbacks.ReduceLROnPlateau(
    monitor=MONITOR, mode=MODE, factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

cb_es = keras.callbacks.EarlyStopping(
    monitor=MONITOR, mode=MODE, patience=12, restore_best_weights=True, verbose=1
)

# TensorBoard 콜백을 일시적으로 비활성화 (경로 문제로 인해)
# callbacks = [cb_best, cb_latest, cb_tb, cb_rlr, cb_es]
callbacks = [cb_best, cb_latest, cb_rlr, cb_es]
print("주의: TensorBoard 콜백이 경로 문제로 인해 비활성화되었습니다.")

print(f"run_dir: {run_dir}")
print(f"timestamp: {timestamp}")
print(f"run_timestamp_dir: {run_timestamp_dir}")
print(f"best_ckpt_path: {best_ckpt_path}")
print(f"latest_weights: {latest_weights}")
print(f"tensorboard log_dir: {log_dir}")
print(f"results_dir: {results_dir}")
print(
    f"runtime args: epochs={RUNTIME_EPOCHS}, batch={RUNTIME_BATCH}, "
    f"lr={RUNTIME_LR}, focal_alpha={RUNTIME_FOCAL_ALPHA}, "
    f"break_weight_scale={RUNTIME_BREAK_WEIGHT_SCALE}"
)


# ============================================================================
# 8) 학습 실행
# ============================================================================

# epochs
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
print(f"\n학습 설정 저장: {config_path}")


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

# 학습 종료 시각/에폭수 업데이트
training_config["training_end_time"] = datetime.datetime.now().isoformat()
training_config["total_epochs_trained"] = len(history.history.get("loss", []))
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

# best 모델 로드 (실패 시 latest weights로 복구)
print(f"Loading best model: {best_ckpt_path}")
try:
    best_model = keras.models.load_model(str(best_ckpt_path), compile=False)
except Exception as e:
    print(f"경고: {e}. best.keras 로드 실패, latest weights로 복구합니다.")
    best_model = build_resnet18_like(input_shape=X_train.shape[1:])
    best_model.load_weights(str(best_ckpt_path).replace(".keras", ".weights.h5").replace("best", "latest"))
    print("복구 완료: latest weights 로드")

# history 저장
history_dict = {}
for key, values in history.history.items():
    history_dict[key] = [float(v) for v in values]

history_path = results_dir / "training_history.json"
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history_dict, f, ensure_ascii=False, indent=2)
print(f"\n학습 히스토리 저장: {history_path}")


# ============================================================================
# 10) 평가 - Classification
# ============================================================================

# 고정 threshold 사용 (--fixed-threshold)
best_th = float(args.fixed_threshold)
print(f"fixed threshold = {best_th:.4f}")

# test 예측
pred_test = best_model.predict(X_test, batch_size=BATCH)
y_prob_test = np.asarray(pred_test).reshape(-1)
y_pred_test = (y_prob_test >= best_th).astype(int)

y_test_cls = y_test[:, 0].astype(int)

# 지표 계산
p = precision_score(y_test_cls, y_pred_test, zero_division=0)
r = recall_score(y_test_cls, y_pred_test, zero_division=0)
f1 = f1_score(y_test_cls, y_pred_test, zero_division=0)
auc = roc_auc_score(y_test_cls, y_prob_test)

print(f"\nbest_th = {best_th:.4f}")
print(f"Precision = {p:.4f} | Recall = {r:.4f} | F1 = {f1:.4f} | ROC-AUC = {auc:.4f}")
print("\nClassification report:\n", classification_report(y_test_cls, y_pred_test, digits=4, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_cls, y_pred_test)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

# 평가 결과 저장
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

# (count) + (percent) 표시
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
print(f"\n평가 결과 저장: {eval_results_path}")

print("\n완료!")
print(f"  베스트 모델: {best_ckpt_path}")
print(f"  결과 폴더: {results_dir}")
print(f"  run timestamp: {timestamp}")

if args.skip_evaluation:
    print("평가 건너뜀 (--skip-evaluation)")
else:
    # ========== 평가 상세: 2. light_models/<run>/evaluation/ 저장 ==========
    detail_dir = run_timestamp_dir / "evaluation"
    detail_dir.mkdir(parents=True, exist_ok=True)
    run_name = timestamp
    print("[EVAL 1/7] 기본 지표 계산")
    print(f"\n{'='*60}\n평가 상세 저장 폴더: {detail_dir}\n{'='*60}")

    X_all, y_all = load_all_data_from_light_train_run(data_run_dir)

    print("[EVAL 2/7] 테스트 데이터 요약")

    y_test_cls = y_test[:, 0].astype(int)
    n_total = int(len(X_test))
    n_normal = int((y_test_cls == 0).sum())
    n_break = int((y_test_cls == 1).sum())
    n_break_true = n_break
    test_data_source_desc = f"1. light_train_data/{data_run_dir.name}/test (기본 test 분할)"

    y_prob = np.asarray(best_model.predict(X_test, batch_size=BATCH)).reshape(-1)
    y_pred = (y_prob >= best_th).astype(int)

    precision = precision_score(y_test_cls, y_pred, zero_division=0)
    recall = recall_score(y_test_cls, y_pred, zero_division=0)
    f1 = f1_score(y_test_cls, y_pred, zero_division=0)
    acc = accuracy_score(y_test_cls, y_pred)
    roc_auc = roc_auc_score(y_test_cls, y_prob)
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_cls, y_prob)
    pr_auc = sk_metrics.auc(rec_curve, prec_curve)
    cm = confusion_matrix(y_test_cls, y_pred)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    val_used = False

    fn_at_best = int(((y_test_cls == 1) & (y_pred == 0)).sum())
    fp_at_best = int(((y_test_cls == 0) & (y_pred == 1)).sum())
    fn_ratio_at_best = fn_at_best / n_break_true if n_break_true else 0
    threshold_sensitivity = [{
        "threshold": round(float(best_th), 4),
        "fn_count": fn_at_best,
        "fn_ratio_pct": round(100 * fn_ratio_at_best, 2),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "fp_count": fp_at_best,
        "note": "fixed_threshold_only",
    }]
    th_sens_path = detail_dir / "threshold_sensitivity.json"
    with open(th_sens_path, "w", encoding="utf-8") as f:
        json.dump({"n_break_true": n_break_true, "rows": threshold_sensitivity}, f, ensure_ascii=False, indent=2)
    print("[EVAL 3/7] Confusion matrix / ROC-PR 저장")

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=11)
    plt.tight_layout()
    cm_path = detail_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rec_curve, prec_curve, label=f"PR (AUC={pr_auc:.4f})", lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = detail_dir / "test_roc_pr_curves.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    th_path = None

    fp = (y_test_cls == 0) & (y_pred == 1)
    fn = (y_test_cls == 1) & (y_pred == 0)
    fp_indices = np.where(fp)[0].tolist()
    fn_indices = np.where(fn)[0].tolist()
    tp_mask = (y_test_cls == 1) & (y_pred == 1)
    tn_mask = (y_test_cls == 0) & (y_pred == 0)
    tp_indices = np.where(tp_mask)[0].tolist()
    tn_indices = np.where(tn_mask)[0].tolist()

    acc_all = precision_all = recall_all = f1_all = roc_auc_all = pr_auc_all = 0.0
    cm_all = np.zeros((2, 2), dtype=int)
    cm_all_norm = cm_all.astype(float)
    cm_all_path = roc_all_path = th_all_path = dist_all_path = tp_tn_all_path = detail_dir / "_.png"
    if X_all is not None and len(X_all) > 0:
        print("[EVAL 4/7] light_train_data 전체 데이터 평가")
        y_prob_all = best_model.predict(X_all, batch_size=BATCH, verbose=0).reshape(-1)
        y_pred_all = (y_prob_all >= best_th).astype(int)
        y_all_cls = y_all.reshape(-1).astype(int)
        acc_all = accuracy_score(y_all_cls, y_pred_all)
        precision_all = precision_score(y_all_cls, y_pred_all, zero_division=0)
        recall_all = recall_score(y_all_cls, y_pred_all, zero_division=0)
        f1_all = f1_score(y_all_cls, y_pred_all, zero_division=0)
        roc_auc_all = roc_auc_score(y_all_cls, y_prob_all)
        fpr_all, tpr_all, _ = roc_curve(y_all_cls, y_prob_all)
        prec_curve_all, rec_curve_all, _ = precision_recall_curve(y_all_cls, y_prob_all)
        pr_auc_all = sk_metrics.auc(rec_curve_all, prec_curve_all)
        cm_all = confusion_matrix(y_all_cls, y_pred_all)
        cm_all_norm = cm_all / (cm_all.sum(axis=1, keepdims=True) + 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cm_all, cmap="Blues")
        axes[0].set_title("Confusion Matrix (Counts) [All Data]")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm_all[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
        axes[1].imshow(cm_all_norm, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Confusion Matrix (Normalized) [All Data]")
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{cm_all[i, j]}\n({cm_all_norm[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=11)
        plt.tight_layout()
        cm_all_path = detail_dir / "all_confusion_matrix.png"
        plt.savefig(cm_all_path, dpi=150, bbox_inches="tight")
        plt.close()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr_all, tpr_all, label=f"ROC (AUC={roc_auc_all:.4f})", lw=2)
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve [All Data]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(rec_curve_all, prec_curve_all, label=f"PR (AUC={pr_auc_all:.4f})", lw=2)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve [All Data]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        roc_all_path = detail_dir / "all_roc_pr_curves.png"
        plt.savefig(roc_all_path, dpi=150, bbox_inches="tight")
        plt.close()
        th_all_path = dist_all_path = tp_tn_all_path = detail_dir / "all_extra.png"
        plt.savefig(th_all_path, dpi=150)
        plt.close()
    else:
        print("[EVAL 4/7] light_train_data 전체 데이터 없음(스킵)")

    misclass_summary = {"false_positive": {"count": len(fp_indices), "indices_sample": fp_indices[:50]}, "false_negative": {"count": len(fn_indices), "indices_sample": fn_indices[:50]}, "true_positive": {"count": len(tp_indices), "indices_sample": tp_indices[:50]}, "true_negative": {"count": len(tn_indices), "indices_sample": tn_indices[:50]}}
    if fp_indices:
        misclass_summary["false_positive"]["prob_mean"] = float(y_prob[fp].mean())
        misclass_summary["false_positive"]["prob_std"] = float(y_prob[fp].std())
    if fn_indices:
        misclass_summary["false_negative"]["prob_mean"] = float(y_prob[fn].mean())
        misclass_summary["false_negative"]["prob_std"] = float(y_prob[fn].std())
    if tp_indices:
        misclass_summary["true_positive"]["prob_mean"] = float(y_prob[tp_mask].mean())
        misclass_summary["true_positive"]["prob_std"] = float(y_prob[tp_mask].std())
    if tn_indices:
        misclass_summary["true_negative"]["prob_mean"] = float(y_prob[tn_mask].mean())
        misclass_summary["true_negative"]["prob_std"] = float(y_prob[tn_mask].std())

    dist_by_class_path = detail_dir / "test_prediction_distribution_by_class.png"
    dist_misclassified_path = detail_dir / "test_prediction_distribution_misclassified.png"
    tp_tn_dist_path = detail_dir / "test_tp_tn_distribution.png"
    for _ in [dist_by_class_path, dist_misclassified_path, tp_tn_dist_path]:
        plt.figure(figsize=(8, 5))
        plt.savefig(_, dpi=150)
        plt.close()
    print("[EVAL 5/7] 평가 리포트/보조 파일 저장")

    eval_report = {
        "model_run": run_name,
        "model_path": str(best_ckpt_path),
        "test_data_source": test_data_source_desc,
        "all_data_source": f"1. light_train_data/{data_run_dir.name}/break_imgs.npy",
        "evaluation_time": datetime.datetime.now().isoformat(),
        "test_samples": n_total,
        "class_distribution": {"normal": int(n_normal), "break": int(n_break)},
        "threshold": float(best_th),
        "threshold_source": "fixed_threshold",
        "classification": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": [[float(x) for x in row] for row in cm_norm.tolist()],
            "all_data_metrics": {
                "accuracy": float(acc_all),
                "precision": float(precision_all),
                "recall": float(recall_all),
                "f1_score": float(f1_all),
                "roc_auc": float(roc_auc_all),
                "pr_auc": float(pr_auc_all),
                "confusion_matrix": cm_all.tolist(),
                "confusion_matrix_normalized": [[float(x) for x in row] for row in cm_all_norm.tolist()],
            },
        },
        "misclassification": misclass_summary,
        "outputs": {
            "test_confusion_matrix": str(cm_path),
            "test_roc_pr_curves": str(roc_path),
            "threshold_sensitivity": str(th_sens_path),
        },
        "threshold_sensitivity": {"n_break_true": n_break_true, "rows": threshold_sensitivity},
    }
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            eval_report["training_config_summary"] = {"data_source_run": cfg.get("data_source_run"), "total_epochs_trained": cfg.get("total_epochs_trained"), "loss": cfg.get("loss"), "input_shape": cfg.get("data", {}).get("input_shape") or cfg.get("model", {}).get("input_shape")}
        except Exception:
            pass
    report_path = detail_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)
    print(f"저장: {report_path}")

    if args.target_pass_mode == "recall_priority":
        overall_pass = (recall >= args.target_recall) and (f1 >= args.target_f1)
    else:
        overall_pass = (precision >= args.target_precision) and (recall >= args.target_recall) and (f1 >= args.target_f1)
    feedback = {"stage": "light_model", "evaluation_dir": str(detail_dir), "model_run": run_name, "pass": bool(overall_pass), "recommended_retrain": bool(not overall_pass), "pass_mode": args.target_pass_mode, "criteria": {"target_precision": float(args.target_precision), "target_recall": float(args.target_recall), "target_f1": float(args.target_f1)}, "actual": {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc), "threshold": float(best_th), "fn_count": len(fn_indices)}}
    feedback_path = detail_dir / "training_feedback.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print(f"저장: {feedback_path}")
    print(f"[feedback] pass={overall_pass} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print("[EVAL 6/7] 베스트 모델 갱신")

    models_base = Path(current_dir) / "2. light_models"
    best_alias_dir = Path(current_dir) / "best_light_model"
    best_candidate = _collect_best_candidate_from_reports(models_base, float(args.best_target_recall), float(args.best_target_accuracy))
    criteria = {
        "target_recall": float(args.best_target_recall),
        "target_accuracy": float(args.best_target_accuracy),
    }
    if best_candidate is None:
        history_path = _append_best_selection_history(
            models_base,
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "skip_no_candidate",
                "criteria": criteria,
            },
        )
        _write_best_model_change_details(
            best_alias_dir=best_alias_dir,
            model_alias="best_light_model",
            action="skip_no_candidate",
            criteria=criteria,
            selected={},
            previous=_load_current_best_metrics(best_alias_dir),
            history_path=history_path,
            reason="evaluation_report 후보 없음",
        )
        print(f"베스트 모델 갱신 건너뜀: 후보가 없습니다. history={history_path}")
    else:
        current_best = _load_current_best_metrics(best_alias_dir)
        should_replace = current_best is None or _rank_key(best_candidate["metrics"], args.best_target_recall, args.best_target_accuracy) > _rank_key(current_best["metrics"], args.best_target_recall, args.best_target_accuracy)
        selected_run_dir = models_base / best_candidate["model_run"]
        if not selected_run_dir.exists():
            history_path = _append_best_selection_history(
                models_base,
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "skip_missing_selected_run_dir",
                    "criteria": criteria,
                    "selected": {
                        "model_run": best_candidate["model_run"],
                        "metrics": best_candidate["metrics"],
                        "report_path": best_candidate["report_path"],
                        "evaluation_time": best_candidate.get("evaluation_time"),
                    },
                    "previous": current_best,
                },
            )
            _write_best_model_change_details(
                best_alias_dir=best_alias_dir,
                model_alias="best_light_model",
                action="skip_missing_selected_run_dir",
                criteria=criteria,
                selected={"model_run": best_candidate["model_run"], "metrics": best_candidate["metrics"]},
                previous=current_best,
                history_path=history_path,
                reason="선정 run 디렉터리가 없음",
            )
            print(f"베스트 모델 갱신 건너뜀: 선정 run 경로가 없습니다. history={history_path}")
        elif should_replace:
            _clear_best_alias_dir(best_alias_dir, preserve_files={"best_model_change_details.log"})
            dst_run_dir = best_alias_dir / best_candidate["model_run"]
            shutil.copytree(selected_run_dir, dst_run_dir)
            selection_payload = {
                "updated_at": datetime.datetime.now().isoformat(),
                "criteria": criteria,
                "selected": {
                    "model_run": best_candidate["model_run"],
                    "metrics": best_candidate["metrics"],
                    "report_path": best_candidate["report_path"],
                    "evaluation_time": best_candidate.get("evaluation_time"),
                },
                "previous": current_best,
            }
            with open(best_alias_dir / "best_model_selection.json", "w", encoding="utf-8") as f:
                json.dump(selection_payload, f, ensure_ascii=False, indent=2)
            history_path = _append_best_selection_history(
                models_base,
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "replace_best_model",
                    **selection_payload,
                },
            )
            _write_best_model_change_details(
                best_alias_dir=best_alias_dir,
                model_alias="best_light_model",
                action="replace_best_model",
                criteria=criteria,
                selected=selection_payload["selected"],
                previous=current_best,
                history_path=history_path,
                reason="후보 모델이 현재 베스트보다 우수",
            )
            print(
                f"best_light_model 갱신: run={best_candidate['model_run']} "
                f"(recall={best_candidate['metrics']['recall']:.4f}, accuracy={best_candidate['metrics']['accuracy']:.4f}) "
                f"| saved_to: {dst_run_dir} | history: {history_path}"
            )
            print("베스트 모델이 새로 등록되었습니다.")
        else:
            history_path = _append_best_selection_history(
                models_base,
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "keep_current_best",
                    "criteria": criteria,
                    "selected": {
                        "model_run": best_candidate["model_run"],
                        "metrics": best_candidate["metrics"],
                        "report_path": best_candidate["report_path"],
                        "evaluation_time": best_candidate.get("evaluation_time"),
                    },
                    "previous": current_best,
                },
            )
            _write_best_model_change_details(
                best_alias_dir=best_alias_dir,
                model_alias="best_light_model",
                action="keep_current_best",
                criteria=criteria,
                selected={"model_run": best_candidate["model_run"], "metrics": best_candidate["metrics"]},
                previous=current_best,
                history_path=history_path,
                reason="현재 베스트 모델 유지",
            )
            print(
                f"best_light_model 유지: current={current_best.get('model_run') if current_best else 'None'} "
                f"candidate={best_candidate['model_run']} | history: {history_path}"
            )
            print("이번 모델은 기존 베스트 모델을 넘지 못했습니다.")

    fn_analysis_path = detail_dir / "fn_reduction_analysis.txt"
    with open(fn_analysis_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "파단 미검출(FN) 감소 분석",
                    f"FN 개수: {len(fn_indices)}",
                    f"사용 threshold: {best_th:.4f}",
                    "threshold_sensitivity.json 참고",
                ]
            )
        )
    summary_path = detail_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            f"Light 모델 상세 평가 요약\n"
            f"모델 run: {run_name}\n"
            f"테스트 샘플: {n_total}\n"
            f"Accuracy: {acc:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1: {f1:.4f}\n"
        )
    print("[EVAL 7/7] 상세 평가 완료")
    print(f"상세 평가 결과: {detail_dir}")

