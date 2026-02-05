#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
4. merge_data 전체에 대해 2차 Hard 모델(13. evaluate_hard_model_2nd)의 검출 결과를
엑셀로 정리해서 make_ai/test 아래에 저장하는 스크립트.

사용 방법(권장 순서):
1) 먼저 13. evaluate_hard_model_2nd.py 를 4. merge_data 전체에 대해 실행
   예) PowerShell:
       cd C:\Users\slhg1\OneDrive\Desktop\SMARTCS_Pole\make_ai
       python "13. evaluate_hard_model_2nd.py" --data-dir "4. merge_data" --no-test-only

   → 최신 2차 Hard 모델(run)을 사용해서
     13. evaluate_hard_model_2nd/<timestamp>/test/predictions.csv 가 생성됨.

2) 그 다음 이 스크립트를 실행
   예)
       python "14. export_2nd_eval_to_excel.py"

   → make_ai/test/hard_2nd_merge_data_predictions.xlsx 로 저장.
"""

import os
import glob
import json
from pathlib import Path

import pandas as pd


def _get_latest_eval_dir(base: Path) -> Path:
    """13. evaluate_hard_model_2nd 아래에서 가장 최근 평가 run 디렉터리 반환."""
    if not base.exists():
        raise FileNotFoundError(f"평가 디렉터리를 찾을 수 없습니다: {base}")
    subs = [d for d in base.iterdir() if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"평가 run 디렉터리가 없습니다: {base}")
    return max(subs, key=lambda d: d.name)


def _build_project_to_server_mapping(current_dir: Path):
    """
    anal2_poles_all_*.json (2. anal_pole_list)에서 프로젝트 -> 서버(main/is/kh) 매핑 생성.
    3.1. check_raw_pole_data_info.py 의 build_project_to_server_mapping 와 동일 개념.
    """
    anal_dir = current_dir / "2. anal_pole_list"
    pattern = str(anal_dir / "anal2_poles_all_*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"경고: anal2_poles_all JSON을 찾을 수 없어 서버 매핑을 생성하지 못했습니다: {pattern}")
        return {}
    anal_file = max(files, key=os.path.getmtime)
    print(f"프로젝트-서버 매핑 로드: {anal_file}")
    with open(anal_file, "r", encoding="utf-8") as f:
        anal_data = json.load(f)
    servers_data = anal_data.get("servers", {})
    project_to_server = {}
    for server, server_info in servers_data.items():
        projects = server_info.get("projects", {})
        if isinstance(projects, dict):
            for proj_name in projects.keys():
                project_to_server[proj_name] = server
    return project_to_server


def main():
    current_dir = Path(__file__).resolve().parent

    # 1) 가장 최근 2차 평가 run 찾기
    eval_base = current_dir / "13. evaluate_hard_model_2nd"
    eval_dir = _get_latest_eval_dir(eval_base)
    # scope 자동 선택: test 우선, 없으면 all
    test_dir = eval_dir / "test"
    all_dir = eval_dir / "all"
    if (test_dir / "predictions.csv").exists():
        scope_dir = test_dir
    elif (all_dir / "predictions.csv").exists():
        scope_dir = all_dir
    else:
        scope_dir = test_dir  # 오류 메시지를 위해 기본값 유지
    csv_path = scope_dir / "predictions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"predictions.csv 를 찾을 수 없습니다: {csv_path}\n"
                                f"※ 먼저 13. evaluate_hard_model_2nd.py 를 실행해 주세요.")

    print(f"최근 2차 평가 run 사용: {eval_dir.name}")
    print(f"읽는 파일: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 2) anal2_poles_all_* 기반 프로젝트 -> 서버(main/is/kh) 매핑 로드
    project_to_server = _build_project_to_server_mapping(current_dir)

    # 3) csv_path에서 서버(main/is/kh), 프로젝트, 전주ID 추출
    def _parse_path_info(p: str):
        try:
            path = Path(p)
            parts = [str(part) for part in path.parts]
            server = project = pole_id = None
            if "4. merge_data" in parts:
                idx = parts.index("4. merge_data")
                # parts[idx+1] = break/normal, parts[idx+2] = 프로젝트명, parts[idx+3] = 전주ID
                if idx + 2 < len(parts):
                    project = parts[idx + 2]
                    server = project_to_server.get(project, None)
                if idx + 3 < len(parts):
                    pole_id = parts[idx + 3]
            return server, project, pole_id
        except Exception:
            return None, None, None

    servers, projects, pole_ids = [], [], []
    for p in df["csv_path"].astype(str):
        s, pj, pid = _parse_path_info(p)
        servers.append(s)
        projects.append(pj)
        pole_ids.append(pid)

    df["server"] = servers
    df["project"] = projects
    df["pole_id"] = pole_ids

    # 4) label(0/1)을 사람이 보기 쉬운 텍스트로 추가
    df["label_text"] = df["label"].map({1: "break", 0: "normal"}).fillna("unknown")

    # 5) 각 축별 "모든 박스"(rank 1~N)를 행(row) 단위로 풀어서 저장
    #    → predictions.csv의 best-only 요약값 대신, pred_boxes.json 에 저장된 전체 후보 박스를 사용.
    all_rows = []
    axes = ["x", "y", "z"]
    for _, r in df.iterrows():
        base = {
            "server": r.get("server"),
            "project": r.get("project"),
            "pole_id": r.get("pole_id"),
            "sample_id": r.get("sample_id"),
            "csv_path": r.get("csv_path"),
            "label": r.get("label"),
            "label_text": r.get("label_text"),
        }
        label_val = int(r.get("label", 0)) if pd.notna(r.get("label")) else 0
        label_dir_name = "break" if label_val == 1 else "normal"
        # 13. evaluate_hard_model_2nd.py 에서 pred_boxes.json 은
        # <eval_dir>/<scope>/info/<break|normal> 아래에 저장된다.
        boxes_dir = scope_dir / "info" / label_dir_name
        boxes_json_path = boxes_dir / f"{base['sample_id']}_pred_boxes.json"
        if not boxes_json_path.exists():
            # 해당 샘플의 박스 JSON이 없으면 스킵
            continue
        try:
            with open(boxes_json_path, "r", encoding="utf-8") as f:
                boxes_json = json.load(f)
        except Exception:
            continue

        axes_dict = boxes_json.get("axes", {})
        for axis in axes:
            if axis not in axes_dict:
                continue
            for entry in axes_dict[axis]:
                row = base.copy()
                row.update(
                    {
                        "axis": axis,
                        "rank": entry.get("rank"),
                        "degree_min": entry.get("degree_min"),
                        "degree_max": entry.get("degree_max"),
                        "height_min": entry.get("height_min"),
                        "height_max": entry.get("height_max"),
                        "score": entry.get("score"),
                        "iou": entry.get("iou"),
                        "inside_data_extent": entry.get("inside_data_extent"),
                        "below_min_size": entry.get("below_min_size"),
                    }
                )
                all_rows.append(row)

    if not all_rows:
        raise RuntimeError("축별 박스를 추출할 수 있는 데이터가 없습니다.")

    df_out = pd.DataFrame(all_rows)

    # 6) make_ai/14. export_2nd_eval_to_excel 아래에 엑셀로 저장 (각 행 = 한 축의 한 박스)
    out_dir = current_dir / "14. export_2nd_eval_to_excel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_2nd_merge_data_predictions.xlsx"

    df_out.to_excel(out_path, index=False)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()

