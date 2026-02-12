#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hard 2차 모델(conf_x/y/z.keras)을 TensorFlow SavedModel로 변환한다.

기본 입력:
- `make_ai/hard_model_2nd_best/checkpoints/conf_x.keras`
- `make_ai/hard_model_2nd_best/checkpoints/conf_y.keras`
- `make_ai/hard_model_2nd_best/checkpoints/conf_z.keras`
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def maybe_run_in_wsl() -> None:
    """
    Windows에서 실행 시 WSL2 스크립트로 위임한다.
    `--local`을 주면 현재 환경에서 직접 실행한다.
    """
    run_local = (
        "--local" in sys.argv
        or "--help" in sys.argv
        or "-h" in sys.argv
        or sys.platform != "win32"
    )
    if run_local:
        if "--local" in sys.argv:
            sys.argv = [x for x in sys.argv if x != "--local"]
        return

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sh_path = script_dir / "export_hard_2nd_conf_to_savedmodel_wsl2.sh"
    if not sh_path.exists():
        sh_path = project_root / "make_ai" / "export_hard_2nd_conf_to_savedmodel_wsl2.sh"
    if not sh_path.exists():
        return

    abs_path = sh_path.resolve()
    drive = abs_path.drive
    if drive:
        wsl_path = "/mnt/" + drive[0].lower() + str(abs_path)[len(drive) :].replace("\\", "/")
    else:
        wsl_path = str(abs_path).replace("\\", "/")

    print(f"WSL2에서 변환 실행: {wsl_path}")
    ret = subprocess.run(["wsl", "bash", wsl_path, *sys.argv[1:]], cwd=str(project_root))
    raise SystemExit(ret.returncode)


def patch_dense_quantization_compat() -> None:
    """
    구버전 Keras가 `quantization_config` 필드를 모를 때를 대비한 호환 패치.
    """

    def _patch(module) -> None:
        try:
            origin = module.Dense.from_config.__func__
        except Exception:
            return

        @classmethod
        def _patched(cls, config):
            config = dict(config)
            config.pop("quantization_config", None)
            return origin(cls, config)

        module.Dense.from_config = _patched

    try:
        import keras.layers.core.dense as dense_mod

        _patch(dense_mod)
    except Exception:
        pass
    try:
        import keras.src.layers.core.dense as dense_src

        _patch(dense_src)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export conf_x/y/z.keras to SavedModel.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="conf_*.keras가 있는 디렉터리 (기본: make_ai/hard_model_2nd_best)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="SavedModel 출력 부모 디렉터리 (기본: source-dir)",
    )
    return parser.parse_args()


def main() -> int:
    maybe_run_in_wsl()
    args = parse_args()

    current_dir = Path(__file__).resolve().parent
    source = (current_dir / args.source_dir).resolve() if args.source_dir else (current_dir / "hard_model_2nd_best")
    checkpoints_dir = source / "checkpoints"
    search_dir = checkpoints_dir if checkpoints_dir.is_dir() else source
    output_parent = (current_dir / args.output_dir).resolve() if args.output_dir else source
    output_parent.mkdir(parents=True, exist_ok=True)

    print(f"입력 디렉터리: {search_dir}")
    print(f"출력 디렉터리: {output_parent}")

    try:
        import tensorflow as tf
    except ImportError as exc:
        print(f"TensorFlow import 실패: {exc}")
        return 1

    patch_dense_quantization_compat()

    exported = 0
    for axis in ("x", "y", "z"):
        keras_path = search_dir / f"conf_{axis}.keras"
        if not keras_path.exists():
            print(f"건너뜀(파일 없음): {keras_path}")
            continue

        out_dir = output_parent / f"conf_{axis}_savedmodel"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"로드: {keras_path}")
        model = tf.keras.models.load_model(str(keras_path), compile=False)
        tf.saved_model.save(model, str(out_dir))
        print(f"저장: {out_dir}")
        exported += 1

    if exported == 0:
        print("conf_x/y/z.keras를 찾지 못했습니다. --source-dir를 확인하세요.")
        return 1

    print("SavedModel 변환 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
