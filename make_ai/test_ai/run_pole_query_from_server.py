#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
서버 접속 → 데이터 가져오기 → 데이터 이상 여부 파악 → 라이트/하드 모델 예측 → MLP 3구간(파단/보류/정상) 판정.

사용: python run_pole_query_from_server.py --server main --pole_id 4114X505
      python run_pole_query_from_server.py --server main --pole_id 4114X505 --project "강원원주-202112"

참고: make_ai/1. get_project_info_list.py, 2. get_anal_pole_list.py, 3. get_raw_pole_data.py
      config.poledb, 4. merge_data, 7/8 라이트 모델, 13. evaluate_hard_model_2nd, weight_test MLP
"""

import sys
import os
import re
import argparse
import shutil
import subprocess
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)         # make_ai
project_root = os.path.dirname(parent_dir)        # SMARTCS_Pole (config 위치)

for p in (project_root, parent_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import poledb as PDB

# 모델·스크립트들은 상위 make_ai 디렉터리(parent_dir)에 있고,
# 입·출력 작업 디렉터리는 test_ai/query_pole_work 아래를 사용한다.
ROOT_DIR = Path(parent_dir)

WORK_DIR = Path(current_dir) / "query_pole_work"
RAW_DIR = WORK_DIR / "3. raw_pole_data"
MERGE_DIR = WORK_DIR / "4. merge_data"


def get_project_for_pole(server: str, pole_id: str) -> str:
    """전주가 속한 프로젝트(groupname) 조회. tb_anal_state 또는 tb_pole 사용."""
    try:
        PDB.poledb_init(server)
        if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
            raise RuntimeError("DB 연결 실패")
        q = "SELECT groupname FROM tb_anal_state WHERE poleid = %s LIMIT 1"
        result = PDB.poledb_conn.do_select_pd(q, [pole_id])
        if result is not None and not result.empty:
            return str(result.iloc[0]["groupname"]).strip()
        q2 = "SELECT groupname FROM tb_pole WHERE poleid = %s LIMIT 1"
        result2 = PDB.poledb_conn.do_select_pd(q2, [pole_id])
        if result2 is not None and not result2.empty:
            return str(result2.iloc[0]["groupname"]).strip()
    except Exception as e:
        raise RuntimeError(f"전주 소속 프로젝트 조회 실패: {e}") from e
    raise ValueError(f"전주 {pole_id}에 해당하는 프로젝트를 찾을 수 없습니다.")


def get_poles_for_project(server: str, project_name: str):
    """
    주어진 프로젝트에서 1/2차 분석 완료된 전주 목록 조회.
    (2. get_anal_pole_list.py의 get_anal2_completed_poles를 재사용)
    """
    import importlib.util

    # 분석 완료 전주 조회 스크립트는 상위 make_ai 디렉터리에 존재
    spec = importlib.util.spec_from_file_location(
        "get_anal_pole_list", ROOT_DIR / "2. get_anal_pole_list.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # PDB.poledb_init(server)는 main 쪽에서 이미 호출
    poles = mod.get_anal2_completed_poles(server, project_name) or []
    # [{"poleid": "...", "breakstate": "B/N"}, ...] 형태
    return poles


def fetch_raw_data(server: str, project_name: str, pole_id: str, anal_result: dict, output_base_dir: Path) -> bool:
    """원시 데이터 조회 후 3. raw_pole_data 형식으로 저장. 이미 있으면 덮어쓰기 위해 해당 전주 디렉터리 삭제 후 저장."""
    category = "break" if anal_result.get("breakstate") == "B" else "normal"
    pole_dir = output_base_dir / category / project_name / pole_id
    if pole_dir.exists():
        shutil.rmtree(pole_dir, ignore_errors=True)
    pole_dir.mkdir(parents=True, exist_ok=True)

    try:
        re_out = PDB.get_meas_result(pole_id, "OUT")
        re_in = PDB.get_meas_result(pole_id, "IN")
    except Exception as e:
        raise RuntimeError(f"측정 결과 조회 실패: {e}") from e

    num_sig_out = re_out.shape[0] if re_out is not None and not re_out.empty else 0
    num_sig_in = re_in.shape[0] if re_in is not None and not re_in.empty else 0

    break_suffix = ""
    if anal_result.get("breakstate") == "B":
        bh, bd = anal_result.get("breakheight"), anal_result.get("breakdegree")
        if bh is not None and bd is not None:
            break_suffix = f"_breakheight_{bh}_breakdegree_{bd}"

    def _f(row, col, alts=None):
        for name in [col] + (alts or []):
            if name in row.index and row.get(name) is not None and (hasattr(row.get(name), "__float__") or isinstance(row.get(name), (int, float))):
                try:
                    return float(row[name])
                except (TypeError, ValueError):
                    pass
        return None

    def _s(row, col, alts=None):
        for name in [col] + (alts or []):
            if name in row.index and row.get(name) is not None:
                return str(row[name])
        return ""

    for kk in range(num_sig_in):
        stype, num = "IN", int(re_in["measno"][kk])
        time = str(re_in["sttime"][kk]).split(" ")[0]
        in_x = PDB.get_meas_data(pole_id, num, stype, "x")
        if in_x is not None and not in_x.empty:
            in_x.to_csv(pole_dir / f"{pole_id}_{num}_{time}_IN_x{break_suffix}.csv", index=False)

    measurements_info = {}
    for kk in range(num_sig_out):
        stype, num = "OUT", int(re_out["measno"][kk])
        time = str(re_out["sttime"][kk]).split(" ")[0]
        out_x = PDB.get_meas_data(pole_id, num, stype, "x")
        out_y = PDB.get_meas_data(pole_id, num, stype, "y")
        out_z = PDB.get_meas_data(pole_id, num, stype, "z")
        if out_x is not None and not out_x.empty:
            out_x.to_csv(pole_dir / f"{pole_id}_{num}_{time}_OUT_x{break_suffix}.csv", index=False)
        if out_y is not None and not out_y.empty:
            out_y.to_csv(pole_dir / f"{pole_id}_{num}_{time}_OUT_y{break_suffix}.csv", index=False)
        if out_z is not None and not out_z.empty:
            out_z.to_csv(pole_dir / f"{pole_id}_{num}_{time}_OUT_z{break_suffix}.csv", index=False)
        meas_info = {
            "measno": num,
            "devicetype": "OUT",
            "stdegree": _f(re_out.iloc[kk], "stdegree", ["stDegree"]),
            "eddegree": _f(re_out.iloc[kk], "eddegree", ["edDegree"]),
            "stheight": _f(re_out.iloc[kk], "stheight", ["stHeight"]),
            "edheight": _f(re_out.iloc[kk], "edheight", ["edHeight"]),
            "sttime": _s(re_out.iloc[kk], "sttime", ["stTime"]),
            "endtime": _s(re_out.iloc[kk], "endtime", ["endtime", "edtime"]),
        }
        measurements_info[f"OUT_{num}"] = meas_info

    info_filename = f"{pole_id}_break_info.json" if anal_result.get("breakstate") == "B" else f"{pole_id}_normal_info.json"
    info_data = {
        "poleid": pole_id,
        "project_name": project_name,
        "breakstate": anal_result.get("breakstate", "N"),
        "breakheight": anal_result.get("breakheight") if anal_result.get("breakstate") == "B" else None,
        "breakdegree": anal_result.get("breakdegree") if anal_result.get("breakstate") == "B" else None,
        "measurements": measurements_info,
    }
    with open(pole_dir / info_filename, "w", encoding="utf-8") as f:
        import json
        json.dump(info_data, f, ensure_ascii=False, indent=2)

    return True


def check_data_quality(raw_pole_dir: Path) -> dict:
    """저장된 원시 데이터 이상 여부: 결측, 행 수, x/y/z 채널 존재 여부."""
    report = {"ok": True, "issues": [], "out_meas": 0, "missing_xyz": [], "row_counts": {}}
    if not raw_pole_dir.exists():
        report["ok"] = False
        report["issues"].append("전주 원시 디렉터리 없음")
        return report

    csv_files = list(raw_pole_dir.glob("*.csv"))
    out_x = [f for f in csv_files if "_OUT_x" in f.name]
    out_y = [f for f in csv_files if "_OUT_y" in f.name]
    out_z = [f for f in csv_files if "_OUT_z" in f.name]

    meas_nums = set()
    for f in out_x:
        m = re.search(r"_(\d+)_[^_]+_OUT_x", f.name)
        if m:
            meas_nums.add(m.group(1))
    report["out_meas"] = len(meas_nums)

    for num in meas_nums:
        fx = next((f for f in out_x if f"_OUT_x" in f.name and f"_{num}_" in f.name), None)
        fy = next((f for f in out_y if f"_OUT_y" in f.name and f"_{num}_" in f.name), None)
        fz = next((f for f in out_z if f"_OUT_z" in f.name and f"_{num}_" in f.name), None)
        if not fx or not fy or not fz:
            report["missing_xyz"].append(f"측정 {num}: x/y/z 중 일부 없음")
            report["ok"] = False
        else:
            import pandas as pd
            for p, key in [(fx, "x"), (fy, "y"), (fz, "z")]:
                try:
                    df = pd.read_csv(p, nrows=1000)
                    report["row_counts"][f"{num}_{key}"] = len(df)
                    if len(df) < 10:
                        report["issues"].append(f"측정 {num} {key}: 행 수 부족 ({len(df)})")
                        report["ok"] = False
                except Exception as e:
                    report["issues"].append(f"측정 {num} {key} 읽기 실패: {e}")
                    report["ok"] = False

    if not report["issues"]:
        report["issues"].append("이상 없음")
    return report


def run_merge_data(pole_dir: Path, output_type_dir: Path) -> int:
    """4. merge_data.process_pole_directory 호출."""
    import importlib.util
    # merge 스크립트는 상위 make_ai 디렉터리에 존재
    spec = importlib.util.spec_from_file_location("merge_data", ROOT_DIR / "4. merge_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_pole_directory(str(pole_dir), str(output_type_dir))


def run_light_predict(csv_paths: list, model_dir: Path) -> dict:
    """CSV 경로별 라이트 모델 파단 확률 반환. { csv_path: prob } """
    import importlib.util
    import numpy as np
    # 라이트 모델 전처리 스크립트는 상위 make_ai 디렉터리에 존재
    spec = importlib.util.spec_from_file_location("set_light_train_data", ROOT_DIR / "6. set_light_train_data.py")
    std = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(std)
    prepare_sequence_from_csv = std.prepare_sequence_from_csv
    resize_img_height = getattr(std, "resize_img_height", lambda img, **kw: img)

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return {p: 0.0 for p in csv_paths}

    if not model_dir.exists():
        return {p: 0.0 for p in csv_paths}
    ckpt = model_dir / "checkpoints" / "best.keras"
    if not ckpt.exists():
        return {p: 0.0 for p in csv_paths}
    model = keras.models.load_model(str(ckpt), compile=False)
    img_h = getattr(std, "img_height", 304)

    out = {}
    for csv_path in csv_paths:
        try:
            r = prepare_sequence_from_csv(csv_path=csv_path, sort_by="height", feature_min_max=None)
            if r is None:
                out[csv_path] = 0.0
                continue
            img, meta = r
            img = resize_img_height(img, target_h=img_h)
            X = np.expand_dims(img.astype(np.float32), axis=0)
            pred = model.predict(X, verbose=0)
            prob = float(pred.ravel()[0]) if pred.size else 0.0
            out[csv_path] = prob
        except Exception:
            out[csv_path] = 0.0
    return out


def run_hard_predict_and_get_scores(merge_data_dir: str, our_csv_paths: list) -> list:
    """
    13. evaluate_hard_model_2nd 실행 후, **이번 실행에서 생성된** predictions.csv에서
    (csv_path, x_score, y_score, z_score) 행 목록 반환.
    - 최신 run을 **수정 시각(mtime)** 기준으로 선택해 우리가 방금 돌린 결과만 사용.
    - predictions에 'query_pole_work' 경로가 없으면 예전 run 결과로 간주하고, 해당 run은 무시.
    - our_csv_paths: 우리 merge 출력 CSV 경로 목록 (매칭 검증용).
    """
    # 13번 평가는 상위 make_ai 디렉터리의 스크립트를 사용
    script_13 = ROOT_DIR / "13. evaluate_hard_model_2nd.py"
    if not script_13.exists():
        print("   경고: 13. evaluate_hard_model_2nd.py 없음")
        return []
    
    # 13.은 10. hard_models_1st와 12. hard_models_2nd가 필요 (학습된 모델)
    first_stage = ROOT_DIR / "10. hard_models_1st"
    second_stage = ROOT_DIR / "12. hard_models_2nd"
    
    # 모델 디렉터리 내 run이 있는지 확인
    first_has_runs = first_stage.exists() and any(first_stage.iterdir())
    second_has_runs = second_stage.exists() and any(second_stage.iterdir())
    
    if not first_has_runs or not second_has_runs:
        print(f"   경고: 하드 모델 디렉터리 또는 학습 run 없음")
        print(f"     - 10. hard_models_1st 존재: {first_stage.exists()}, run 있음: {first_has_runs}")
        print(f"     - 12. hard_models_2nd 존재: {second_stage.exists()}, run 있음: {second_has_runs}")
        print(f"   → 하드 모델을 먼저 학습(10., 12. 스크립트)해야 합니다. x/y/z=0으로 진행.")
        return []
    # Windows에서 13.은 기본적으로 WSL로 넘어가며, WSL에 venv_wsl2 없으면 실패함.
    # query_pole_work 데이터는 Windows 경로에 있으므로 --local 로 Windows에서 직접 실행.
    # --repreprocess: 캐시 사용 안 함 (query_pole_work 경로 캐시 디렉터리 생성 오류 방지)
    cmd = [
        sys.executable,
        str(script_13),
        "--local",
        "--data-dir",
        merge_data_dir,
        "--no-test-only",
        # max-files=0 → 13번 스크립트에서 제한 없이 전체 파일 사용
        "--max-files",
        "0",
        "--max-points",
        "2000",
        "--repreprocess",
    ]
    try:
        # 13.의 캐시 저장 시 필요한 서브디렉터리 미리 생성
        # (cache_key에 경로 구분자가 포함되어 query_pole_work 서브디렉터리 필요)
        hard_data_base = ROOT_DIR / "9. hard_train_data"
        if hard_data_base.exists():
            for run_dir in hard_data_base.iterdir():
                if run_dir.is_dir():
                    eval_cache = run_dir / "eval_cache"
                    if eval_cache.exists():
                        query_cache = eval_cache / "query_pole_work"
                        query_cache.mkdir(parents=True, exist_ok=True)

        # CPU만 사용하도록 환경변수 설정 (GPU 비활성화)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow 경고 줄임

        print("   [13] 하드 모델 평가 시작 (실시간 로그 표시 중...)")
        # stdout/stderr를 캡처하지 않고 그대로 터미널로 흘려보냄
        # conf bin 이미지까지 생성하려면 오래 걸릴 수 있으므로 타임아웃을 넉넉하게(예: 3600초) 설정
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            timeout=3600,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            print(f"   [13] 종료 코드: {result.returncode}")
        else:
            print("   [13] 평가 완료")
    except subprocess.TimeoutExpired:
        print("   경고: 13. evaluate_hard_model_2nd 타임아웃(600초)")
    except Exception as e:
        print("   경고: 13. 실행 예외:", e)

    eval_base = ROOT_DIR / "13. evaluate_hard_model_2nd"
    if not eval_base.exists():
        print(f"   [디버그] 13. evaluate_hard_model_2nd 출력 디렉터리 없음: {eval_base}")
        return []
    # 수정 시각 기준 최신 run 사용 (방금 실행한 결과가 이 run이어야 함)
    runs = sorted(
        [d for d in eval_base.iterdir() if d.is_dir()],
        key=lambda d: os.path.getmtime(d) if d.exists() else 0,
        reverse=True,
    )
    print(f"   [디버그] 13. run 디렉터리 수: {len(runs)}, 최신: {runs[0].name if runs else 'None'}")
    
    for idx, run_dir in enumerate(runs):
        for scope in ("all", "test"):
            csv_file = run_dir / scope / "predictions.csv"
            if not csv_file.exists():
                if idx == 0:  # 최신 run만 로그
                    print(f"   [디버그] {run_dir.name}/{scope}/predictions.csv 없음")
                continue
            print(f"   [디버그] {run_dir.name}/{scope}/predictions.csv 발견 (파일 크기: {csv_file.stat().st_size} bytes)")
            import pandas as pd
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            rows = []
            for _, row in df.iterrows():
                csv_path = row.get("csv_path", "")
                x_score = float(row.get("x_score", 0) or 0)
                y_score = float(row.get("y_score", 0) or 0)
                z_score = float(row.get("z_score", 0) or 0)
                rows.append({"csv_path": csv_path, "x_score": x_score, "y_score": y_score, "z_score": z_score, "label": row.get("label", 0)})
            # 우리 데이터(query_pole_work) 결과만 사용: 예전 4. merge_data run과 혼동 방지
            if not rows:
                print(f"   [디버그] predictions.csv 행 없음")
                continue
            print(f"   [디버그] predictions.csv 행 수: {len(rows)}")
            query_pole_paths = [r for r in rows if "query_pole_work" in str(r.get("csv_path", ""))]
            print(f"   [디버그] 'query_pole_work' 포함 경로 수: {len(query_pole_paths)}")
            if query_pole_paths:
                print(f"   [디버그] 예시 경로: {query_pole_paths[0].get('csv_path', '')[:100]}")
            any_our_path = len(query_pole_paths) > 0
            if any_our_path:
                print(f"   [디버그] query_pole_work 결과 {len(query_pole_paths)}개 사용")
                return rows
            # 첫 run이 우리 run이 아니면(경로에 query_pole_work 없음) 다음 run 시도하지 않고 빈 결과 반환
            # (예전에 4. merge_data로 돌린 run이 최신 mtime일 수 있음)
            print(f"   [디버그] 최신 run에 query_pole_work 경로 없음. 예전 run으로 간주하고 빈 결과 반환.")
            return []
    print(f"   [디버그] 모든 run에서 predictions.csv 없음")
    return []


def run_mlp_3zone(light_prob: float, max_x: float, max_y: float, max_z: float, model_path: Path, thresholds_path: Path) -> tuple:
    """MLP 확률 및 3구간 판정(정상/보류/파단) 반환."""
    try:
        from tensorflow import keras
    except ImportError:
        return 0.0, "정상"

    thresh_normal = 0.65
    thresh_pardan = 0.95
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                key, val = line.split("\t", 1)
                if key == "정상_상한":
                    thresh_normal = float(val)
                elif key == "파단_하한":
                    thresh_pardan = float(val)

    model = keras.models.load_model(str(model_path), compile=False)
    import numpy as np
    X = np.array([[light_prob, max_x, max_y, max_z]], dtype=np.float32)
    proba = float(model.predict(X, verbose=0).ravel()[0])
    if proba < thresh_normal:
        verdict = "정상"
    elif proba < thresh_pardan:
        verdict = "보류"
    else:
        verdict = "파단"
    return proba, verdict
    return proba, verdict


def run_for_single_pole(server: str, pole_id: str, project_name: str):
    """단일 전주에 대해 전체 파이프라인 실행하고, 측정별 결과 리스트 반환."""
    # query_pole_work 디렉터리 초기화 (이전 데이터 제거)
    if WORK_DIR.exists():
        print(f"이전 작업 디렉터리 삭제: {WORK_DIR}")
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. 서버 접속 및 전주 소속 프로젝트 조회")
    print("=" * 60)
    PDB.poledb_init(server)
    if project_name is None:
        project_name = get_project_for_pole(server, pole_id)
    print(f"   서버: {server}, 전주: {pole_id}, 프로젝트: {project_name}")

    # 2차 분석 결과(B/N) 조회
    import importlib.util

    # 원시 데이터 조회 스크립트는 상위 make_ai 디렉터리에 존재
    spec_raw = importlib.util.spec_from_file_location(
        "get_raw_pole_data", ROOT_DIR / "3. get_raw_pole_data.py"
    )
    raw_mod = importlib.util.module_from_spec(spec_raw)
    spec_raw.loader.exec_module(raw_mod)
    anal_result = raw_mod.get_pole_anal2_result(server, project_name, pole_id)
    if anal_result is None:
        print("   오류: 해당 전주의 2차 분석 결과(B/N)를 찾을 수 없습니다.")
        return project_name, []
    print(
        f"   분석 결과: {anal_result.get('breakstate', 'N')} "
        f"(breakheight={anal_result.get('breakheight')}, breakdegree={anal_result.get('breakdegree')})"
    )

    print("\n2. 데이터 가져오기 (원시 저장)")
    print("-" * 60)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_raw_data(server, project_name, pole_id, anal_result, RAW_DIR)
    category = "break" if anal_result.get("breakstate") == "B" else "normal"
    pole_raw_dir = RAW_DIR / category / project_name / pole_id
    print(f"   저장 경로: {pole_raw_dir}")

    print("\n3. 데이터 이상 여부 파악")
    print("-" * 60)
    quality = check_data_quality(pole_raw_dir)
    for msg in quality["issues"]:
        print(f"   {msg}")
    print(
        f"   OUT 측정 수: {quality['out_meas']}, 이상 여부: "
        f"{'정상' if quality['ok'] else '주의'}"
    )

    print("\n4. 전처리 (merge_data)")
    print("-" * 60)
    type_out = MERGE_DIR / category
    type_out.mkdir(parents=True, exist_ok=True)
    cnt = run_merge_data(pole_raw_dir, type_out)
    print(f"   처리된 CSV 수: {cnt}")
    if cnt == 0:
        print("   오류: merge_data 출력이 없어 예측을 진행할 수 없습니다.")
        return project_name, []

    csv_list = list(
        (MERGE_DIR / category / project_name / pole_id).glob("*_OUT_processed.csv")
    )
    csv_paths = [str(p) for p in csv_list]
    if not csv_paths:
        print("   오류: 처리된 CSV 파일을 찾을 수 없습니다.")
        return project_name, []

    print("\n5. 라이트 모델 예측")
    print("-" * 60)
    # 라이트 모델 디렉터리는 상위 make_ai 기준
    light_models_base = ROOT_DIR / "7. light_models"
    latest_light = None
    if light_models_base.exists():
        for d in sorted(light_models_base.iterdir(), key=lambda x: x.name, reverse=True):
            if d.is_dir() and (d / "checkpoints" / "best.keras").exists():
                latest_light = d
                break
    if latest_light is None:
        print("   경고: 7. light_models 최신 run 없음. 라이트 확률 0으로 진행.")
    else:
        print(f"   라이트 모델: {latest_light.name}")
    light_probs = (
        run_light_predict(csv_paths, latest_light)
        if latest_light
        else {p: 0.0 for p in csv_paths}
    )

    print("\n6. 하드 모델 예측 (13. evaluate_hard_model_2nd 실행)")
    print("-" * 60)
    # 13번 스크립트는 make_ai 기준으로 data-dir을 해석하므로
    # test_ai/query_pole_work 경로를 상대 경로로 넘긴다.
    merge_rel = os.path.join("test_ai", "query_pole_work", "4. merge_data")
    hard_rows = run_hard_predict_and_get_scores(merge_rel, csv_paths)
    if not hard_rows:
        print(
            "   경고: 하드 예측 결과 없음(13. 실행 실패 또는 query_pole_work 결과 없음). "
            "x/y/z score 0으로 진행."
        )
        print(
            "   참고: 13. evaluate_hard_model_2nd가 query_pole_work/4. merge_data 데이터로 "
            "결과를 내려면, 해당 경로에 break 또는 normal/프로젝트/전주/*_OUT_processed.csv 가 "
            "있어야 합니다."
        )

    # csv_path 매칭: 1) norm 경로 2) 파일명(basename) fallback
    def norm_path(p):
        return str(Path(p).resolve()) if p else ""

    csv_to_light = {norm_path(k): v for k, v in light_probs.items()}
    hard_by_path = {}
    for r in hard_rows:
        key = norm_path(r.get("csv_path", ""))
        if key:
            hard_by_path[key] = r
    hard_by_basename = {
        Path(r.get("csv_path", "")).name: r for r in hard_rows if r.get("csv_path")
    }
    for p in csv_paths:
        key = norm_path(p)
        if key in hard_by_path:
            continue
        basename = Path(p).name
        if basename in hard_by_basename:
            hard_by_path[key] = hard_by_basename[basename]
        else:
            hard_by_path[key] = {
                "csv_path": p,
                "x_score": 0.0,
                "y_score": 0.0,
                "z_score": 0.0,
                "label": 0,
            }

    print("\n7. MLP 3구간 판정 (파단/보류/정상)")
    print("-" * 60)
    # MLP 모델은 상위 make_ai/weight_num 경로 사용
    wt_dir = ROOT_DIR / "weight_num" / "export_2nd_eval_to_excel"
    mlp_path = wt_dir / "eval_mlp_model.keras"
    th_path = wt_dir / "eval_mlp_3zone_thresholds.txt"
    if not mlp_path.exists():
        print(
            "   경고: MLP 모델 없음 "
            "(weight_test/export_2nd_eval_to_excel/eval_mlp_model.keras). "
            "판정은 확률만 출력."
        )

    results = []
    for i, csv_path in enumerate(csv_paths):
        key = norm_path(csv_path)
        lp = csv_to_light.get(key, 0.0)
        hr = hard_by_path.get(key, {})
        mx, my, mz = (
            hr.get("x_score", 0.0),
            hr.get("y_score", 0.0),
            hr.get("z_score", 0.0),
        )
        mesonu = (
            Path(csv_path).stem.replace("_OUT_processed", "").split("_")[-1]
            if "_" in Path(csv_path).stem
            else str(i + 1)
        )
        if mlp_path.exists():
            proba, verdict = run_mlp_3zone(lp, mx, my, mz, mlp_path, th_path)
        else:
            proba, verdict = 0.0, "-"
        results.append(
            {
                "project": project_name,
                "pole_id": pole_id,
                "csv_path": csv_path,
                "측정번호": mesonu,
                "라이트_확률": round(lp, 4),
                "x_score": round(mx, 4),
                "y_score": round(my, 4),
                "z_score": round(mz, 4),
                "MLP_확률": round(proba, 4),
                "판정": verdict,
            }
        )
        print(
            f"   측정 {mesonu}: light={lp:.3f}, x={mx:.3f}, y={my:.3f}, z={mz:.3f} "
            f"-> MLP={proba:.3f} -> {verdict}"
        )

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    for r in results:
        print(f"   측정번호 {r['측정번호']}: {r['판정']} (MLP={r['MLP_확률']})")

    return project_name, results


def main():
    parser = argparse.ArgumentParser(
        description="서버 접속 → 데이터 수집 → 이상 검사 → 라이트/하드 예측 → MLP 파단/보류/정상"
    )
    parser.add_argument("--server", required=True, help="서버 이름 (main, is, kh 등)")
    parser.add_argument("--pole_id", required=False, default=None, help="전주 ID")
    parser.add_argument(
        "--project", default=None, help="프로젝트명 (미지정 시 단일 전주 모드에서 DB 조회)"
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="해당 프로젝트의 모든 전주를 배치로 예측 (pole_id 없이 사용)",
    )
    args = parser.parse_args()

    server = args.server.strip()
    pole_id = args.pole_id.strip() if args.pole_id else None
    project_name = args.project.strip() if args.project else None

    # 1) 프로젝트 배치 모드: --project-only 또는 pole_id 미지정 + project 지정
    if (args.project_only or pole_id is None) and project_name:
        from datetime import datetime
        import pandas as pd
        import importlib.util

        PDB.poledb_init(server)
        poles = get_poles_for_project(server, project_name)
        if not poles:
            print(f"프로젝트 '{project_name}'에 대해 예측할 전주 목록을 찾지 못했습니다.")
            return 1

        print("=" * 60)
        print(f"프로젝트 배치 예측 시작: 서버={server}, 프로젝트={project_name}, 전주 수={len(poles)}")
        print("=" * 60)

        # 배치 시작 전에 test_ai/query_pole_work 전체 초기화
        if WORK_DIR.exists():
            print(f"기존 배치 작업 디렉터리 삭제: {WORK_DIR}")
            shutil.rmtree(WORK_DIR)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # 1단계: 모든 전주에 대해 raw 저장 + merge_data 전처리만 수행
        spec_raw = importlib.util.spec_from_file_location(
            "get_raw_pole_data", ROOT_DIR / "3. get_raw_pole_data.py"
        )
        raw_mod = importlib.util.module_from_spec(spec_raw)
        spec_raw.loader.exec_module(raw_mod)

        for info in poles:
            pid = info.get("poleid")
            print("\n" + "#" * 60)
            print(f"[1/2] 전주 {pid} 데이터 수집·전처리 중...")
            print("#" * 60)

            anal_result = raw_mod.get_pole_anal2_result(server, project_name, pid)
            if anal_result is None:
                print(f"   경고: 전주 {pid}의 2차 분석 결과(B/N)를 찾을 수 없습니다. 스킵.")
                continue

            fetch_raw_data(server, project_name, pid, anal_result, RAW_DIR)
            category = "break" if anal_result.get("breakstate") == "B" else "normal"
            pole_raw_dir = RAW_DIR / category / project_name / pid
            print(f"   저장 경로: {pole_raw_dir}")

            quality = check_data_quality(pole_raw_dir)
            for msg in quality["issues"]:
                print(f"   {msg}")
            print(
                f"   OUT 측정 수: {quality['out_meas']}, 이상 여부: "
                f"{'정상' if quality['ok'] else '주의'}"
            )

            type_out = MERGE_DIR / category
            type_out.mkdir(parents=True, exist_ok=True)
            cnt = run_merge_data(pole_raw_dir, type_out)
            print(f"   처리된 CSV 수: {cnt}")

        # 2단계: 모든 전주의 merge_data 결과를 한 번에 예측
        print("\n" + "#" * 60)
        print(f"[2/2] 모든 전주 merge_data 결과에 대해 예측 수행 중...")
        print("#" * 60)

        csv_paths: list[str] = []
        for category in ("break", "normal"):
            base = MERGE_DIR / category / project_name
            if not base.exists():
                continue
            # project_name 아래의 모든 전주 디렉터리 내부 *_OUT_processed.csv
            for p in base.rglob("*_OUT_processed.csv"):
                csv_paths.append(str(p))

        if not csv_paths:
            print("예측에 사용할 merge_data CSV 파일이 없습니다.")
            return 1

        # 라이트 모델 예측
        print("\n5. 라이트 모델 예측 (프로젝트 전체)")
        print("-" * 60)
        light_models_base = ROOT_DIR / "7. light_models"
        latest_light = None
        if light_models_base.exists():
            for d in sorted(light_models_base.iterdir(), key=lambda x: x.name, reverse=True):
                if d.is_dir() and (d / "checkpoints" / "best.keras").exists():
                    latest_light = d
                    break
        if latest_light is None:
            print("   경고: 7. light_models 최신 run 없음. 라이트 확률 0으로 진행.")
        else:
            print(f"   라이트 모델: {latest_light.name}")
        light_probs = (
            run_light_predict(csv_paths, latest_light)
            if latest_light
            else {p: 0.0 for p in csv_paths}
        )

        # 하드 모델 예측 (13번 한 번 호출)
        print("\n6. 하드 모델 예측 (13. evaluate_hard_model_2nd 실행, 프로젝트 전체)")
        print("-" * 60)
        merge_rel = os.path.join("test_ai", "query_pole_work", "4. merge_data")
        hard_rows = run_hard_predict_and_get_scores(merge_rel, csv_paths)
        if not hard_rows:
            print(
                "   경고: 하드 예측 결과 없음(13. 실행 실패 또는 query_pole_work 결과 없음). "
                "x/y/z score 0으로 진행."
            )

        def norm_path(p):
            return str(Path(p).resolve()) if p else ""

        csv_to_light = {norm_path(k): v for k, v in light_probs.items()}
        hard_by_path: dict[str, dict] = {}
        no_hard_keys: set[str] = set()
        for r in hard_rows:
            key = norm_path(r.get("csv_path", ""))
            if key:
                hard_by_path[key] = r
        hard_by_basename = {
            Path(r.get("csv_path", "")).name: r for r in hard_rows if r.get("csv_path")
        }
        for p in csv_paths:
            key = norm_path(p)
            if key in hard_by_path:
                continue
            basename = Path(p).name
            if basename in hard_by_basename:
                hard_by_path[key] = hard_by_basename[basename]
            else:
                hard_by_path[key] = {
                    "csv_path": p,
                    "x_score": 0.0,
                    "y_score": 0.0,
                    "z_score": 0.0,
                    "label": 0,
                }
                no_hard_keys.add(key)

        # MLP 3구간 판정
        print("\n7. MLP 3구간 판정 (파단/보류/정상, 프로젝트 전체)")
        print("-" * 60)
        wt_dir = ROOT_DIR / "weight_num" / "export_2nd_eval_to_excel"
        mlp_path = wt_dir / "eval_mlp_model.keras"
        th_path = wt_dir / "eval_mlp_3zone_thresholds.txt"
        if not mlp_path.exists():
            print(
                "   경고: MLP 모델 없음 "
                "(weight_test/export_2nd_eval_to_excel/eval_mlp_model.keras). "
                "판정은 확률만 출력."
            )

        all_rows = []
        for i, csv_path in enumerate(csv_paths):
            key = norm_path(csv_path)
            lp = csv_to_light.get(key, 0.0)
            hr = hard_by_path.get(key, {})
            mx, my, mz = (
                hr.get("x_score", 0.0),
                hr.get("y_score", 0.0),
                hr.get("z_score", 0.0),
            )
            stem = Path(csv_path).stem.replace("_OUT_processed", "")
            parts = stem.split("_")
            pole_id = parts[0] if parts else ""
            mesonu = parts[-1] if len(parts) >= 2 else str(i + 1)
            if mlp_path.exists():
                proba, verdict = run_mlp_3zone(lp, mx, my, mz, mlp_path, th_path)
            else:
                proba, verdict = 0.0, "-"

            # 하드 점수 0의 원인 표시
            if not hard_rows:
                hard_reason = "hard_eval_failed_or_no_data"
            elif key in no_hard_keys:
                hard_reason = "no_hard_prediction_for_csv"
            else:
                hard_reason = "hard_ok"

            all_rows.append(
                {
                    "project": project_name,
                    "pole_id": pole_id,
                    "csv_path": csv_path,
                    "측정번호": mesonu,
                    "라이트_확률": round(lp, 4),
                    "x_score": round(mx, 4),
                    "y_score": round(my, 4),
                    "z_score": round(mz, 4),
                    "MLP_확률": round(proba, 4),
                    "판정": verdict,
                    "hard_reason": hard_reason,
                }
            )

        if not all_rows:
            print("예측 결과가 없습니다.")
            return 1

        df = pd.DataFrame(all_rows)

        # 전주 단위 요약 시트용 데이터 생성
        def _summarize_pole(group: pd.DataFrame) -> pd.Series:
            project = group["project"].iloc[0] if "project" in group.columns else ""
            max_mlp = float(group["MLP_확률"].max()) if "MLP_확률" in group.columns else 0.0
            verdict = "정상"
            if "판정" in group.columns:
                if (group["판정"] == "파단").any():
                    verdict = "파단"
                elif (group["판정"] == "보류").any():
                    verdict = "보류"
                else:
                    verdict = "정상"
            return pd.Series(
                {
                    "project": project,
                    "pole_verdict": verdict,
                    "max_MLP_확률": round(max_mlp, 4),
                }
            )

        by_pole = df.groupby("pole_id", as_index=False).apply(_summarize_pole)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = WORK_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"batch_predictions_{server}_{project_name}_{timestamp}.xlsx"

        # 시트 2개: 측정 단위(by_measure), 전주 단위(by_pole)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="by_measure")
            by_pole.to_excel(writer, index=False, sheet_name="by_pole")

        print(f"\n배치 예측 결과 엑셀 저장: {out_path}")

        # 전주별 최대 MLP 확률 기준 상위 10개 전주 출력
        pole_max = (
            by_pole[["pole_id", "max_MLP_확률"]]
            .sort_values("max_MLP_확률", ascending=False)
        )
        top10 = pole_max.head(10)
        print("\nMLP 값이 높은 상위 10개 전주:")
        print("-" * 60)
        for _, row in top10.iterrows():
            print(f"전주 {row['pole_id']}: max MLP={row['max_MLP_확률']:.4f}")
        return 0

    # 2) 기존 단일 전주 모드 (pole_id 필수)
    if not pole_id:
        print("단일 전주 모드에서는 --pole_id 를 지정해야 합니다.")
        return 1

    _, _ = run_for_single_pole(server, pole_id, project_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
