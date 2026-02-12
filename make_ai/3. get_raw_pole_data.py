#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""분석 완료 전주 목록을 기준으로 원본 계측 CSV를 내려받아 저장한다."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import poledb as PDB


SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}
DEFAULT_INPUT_DIR = CURRENT_DIR / "2. anal_pole_list"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "3. raw_pole_data"


def find_latest_json(input_dir: Path) -> Path:
    """`anal2_poles_all_*.json` 중 최신 파일을 찾는다."""
    files = glob.glob(str(input_dir / "anal2_poles_all_*.json"))
    if not files:
        raise FileNotFoundError(f"anal2_poles_all_*.json 파일이 없습니다: {input_dir}")
    return Path(max(files, key=lambda p: Path(p).stat().st_mtime))


def load_input(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def query_latest_anal_result(project_name: str, poleid: str) -> Optional[Dict[str, Optional[float]]]:
    """전주의 최신(2차 우선) 분석 결과를 조회한다."""
    if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
        return None

    query = """
        SELECT
            tar.poleid,
            COALESCE(tar.breakstate, 'N') AS breakstate,
            tar.breakheight,
            tar.breakdegree,
            tas.groupname
        FROM tb_anal_result tar
        INNER JOIN tb_anal_state tas ON tar.poleid = tas.poleid
        INNER JOIN (
            SELECT poleid, MAX(analstep) AS max_analstep
            FROM tb_anal_result
            WHERE analstep IN (1, 2)
            GROUP BY poleid
        ) max_step ON tar.poleid = max_step.poleid AND tar.analstep = max_step.max_analstep
        INNER JOIN (
            SELECT poleid, analstep, MAX(regdate) AS max_regdate
            FROM tb_anal_result
            WHERE analstep IN (1, 2)
            GROUP BY poleid, analstep
        ) latest
          ON tar.poleid = latest.poleid
         AND tar.analstep = latest.analstep
         AND tar.regdate = latest.max_regdate
        WHERE tas.groupname = %s
          AND tar.poleid = %s
        LIMIT 1
    """
    result = PDB.poledb_conn.do_select_pd(query, [project_name, poleid])
    if result is None or result.empty:
        return None

    row = result.iloc[0]
    if str(row.get("groupname", "")).strip() != project_name:
        return None
    if str(row.get("poleid", "")).strip().upper() != str(poleid).strip().upper():
        return None

    breakstate = str(row.get("breakstate", "N")).strip().upper()
    if breakstate not in {"B", "N"}:
        breakstate = "N"
    return {
        "breakstate": breakstate,
        "breakheight": safe_float(row.get("breakheight")) if breakstate == "B" else None,
        "breakdegree": safe_float(row.get("breakdegree")) if breakstate == "B" else None,
    }


def pole_has_csvs(pole_dir: Path) -> bool:
    return pole_dir.exists() and any(p.suffix.lower() == ".csv" for p in pole_dir.iterdir())


def get_saved_pole_ids(base_dir: Path, project_name: str, category: str) -> Set[str]:
    """이미 저장된 전주 ID를 반환한다."""
    project_dir = base_dir / category / project_name
    if not project_dir.exists():
        return set()
    return {p.name for p in project_dir.iterdir() if p.is_dir() and pole_has_csvs(p)}


def save_raw_measurements(
    project_name: str,
    poleid: str,
    anal_result: Dict[str, Optional[float]],
    output_base: Path,
) -> bool:
    """단일 전주의 원본 계측 데이터를 저장한다."""
    category = "break" if anal_result["breakstate"] == "B" else "normal"
    target_dir = output_base / category / project_name / poleid
    target_dir.mkdir(parents=True, exist_ok=True)

    if pole_has_csvs(target_dir):
        return True

    re_out = PDB.get_meas_result(poleid, "OUT")
    re_in = PDB.get_meas_result(poleid, "IN")

    def save_axis_csv(df: pd.DataFrame, devicetype: str, axis: str, idx: int) -> None:
        measno = int(df["measno"][idx])
        dt = str(df["sttime"][idx]).split(" ")[0]
        meas_df = PDB.get_meas_data(poleid, measno, devicetype, axis)
        if meas_df is None or meas_df.empty:
            return
        suffix = ""
        if anal_result["breakstate"] == "B":
            bh = anal_result.get("breakheight")
            bd = anal_result.get("breakdegree")
            if bh is not None and bd is not None:
                suffix = f"_breakheight_{bh}_breakdegree_{bd}"
        out_name = f"{poleid}_{idx+1}_{dt}_{devicetype}_{axis}{suffix}.csv"
        meas_df.to_csv(target_dir / out_name, index=False)

    out_count = 0
    if re_in is not None and not re_in.empty:
        for idx in range(len(re_in)):
            save_axis_csv(re_in, "IN", "x", idx)
            out_count += 1

    if re_out is not None and not re_out.empty:
        for idx in range(len(re_out)):
            for axis in ("x", "y", "z"):
                save_axis_csv(re_out, "OUT", axis, idx)
            out_count += 1

    info_name = f"{poleid}_break_info.json" if anal_result["breakstate"] == "B" else f"{poleid}_normal_info.json"
    info_data = {
        "poleid": poleid,
        "project_name": project_name,
        "breakstate": anal_result["breakstate"],
        "breakheight": anal_result.get("breakheight"),
        "breakdegree": anal_result.get("breakdegree"),
    }
    with (target_dir / info_name).open("w", encoding="utf-8") as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)

    return out_count > 0


def process_category(
    servers_data: Dict,
    output_dir: Path,
    stats: Dict[str, int],
    category: str,
    normal_limit: Optional[int] = None,
    already_saved_normal: int = 0,
) -> int:
    """카테고리(break/normal)별 데이터를 수집한다."""
    expected_breakstate = "B" if category == "break" else "N"
    collected_normal = already_saved_normal

    for server, server_payload in servers_data.items():
        if server not in SERVERS:
            continue

        print(f"\n[{SERVERS[server]}] {category} 수집 시작")
        PDB.poledb_init(server)
        projects = server_payload.get("projects", {})
        for project_name, project_info in projects.items():
            if normal_limit is not None and collected_normal >= normal_limit:
                break

            target_ids = project_info.get("pole_ids", [])
            saved_ids = get_saved_pole_ids(output_dir, project_name, category)
            pending_ids = [pid for pid in target_ids if pid not in saved_ids]
            if not pending_ids:
                continue

            for poleid in tqdm(pending_ids, desc=f"  {project_name}", unit="pole", leave=False):
                if normal_limit is not None and collected_normal >= normal_limit:
                    break

                result = query_latest_anal_result(project_name, poleid)
                if result is None:
                    stats["skipped"] += 1
                    continue
                if result["breakstate"] != expected_breakstate:
                    continue

                ok = save_raw_measurements(project_name, poleid, result, output_dir)
                if not ok:
                    stats["errors"] += 1
                    continue

                if category == "break":
                    stats["saved_break"] += 1
                else:
                    stats["saved_normal"] += 1
                    collected_normal += 1

    return collected_normal


def count_total_saved(output_dir: Path) -> Tuple[int, int]:
    """누적 저장된 break/normal 전주 수를 계산한다."""
    break_count = 0
    normal_count = 0

    for category, ref in (("break", "break"), ("normal", "normal")):
        base = output_dir / category
        if not base.exists():
            continue
        for project_dir in base.iterdir():
            if not project_dir.is_dir():
                continue
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                if pole_has_csvs(pole_dir):
                    if ref == "break":
                        break_count += 1
                    else:
                        normal_count += 1
    return break_count, normal_count


def save_summary(
    output_dir: Path,
    source_json: Path,
    stats: Dict[str, int],
    normal_limit: int,
    total_break: int,
    total_normal: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = output_dir / f"raw_pole_data_summary_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "source_json": source_json.name,
                "source_path": str(source_json),
                "output_dir": str(output_dir),
                "stats": stats,
                "normal_ratio_limit": normal_limit,
                "total_saved_break": total_break,
                "total_saved_normal": total_normal,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본 전주 데이터 다운로드")
    parser.add_argument(
        "--input-json",
        default=None,
        help="입력 anal2_poles_all_*.json 경로 (미지정 시 최신 파일 자동 탐색)",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="입력 JSON 탐색 디렉터리",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="원본 데이터 저장 디렉터리",
    )
    parser.add_argument(
        "--normal-ratio",
        type=int,
        default=10,
        help="정상 전주 최대 비율 = 파단 전주 수 * normal-ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json) if args.input_json else find_latest_json(Path(args.input_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_input(input_json)
    servers_data = payload.get("servers", {})
    stats = {
        "saved_break": 0,
        "saved_normal": 0,
        "skipped": 0,
        "errors": 0,
    }

    print("=" * 70)
    print("3단계: 원본 데이터 수집 시작")
    print(f"입력 JSON: {input_json}")
    print("=" * 70)

    process_category(servers_data, output_dir, stats, category="break")
    total_break, total_normal = count_total_saved(output_dir)
    normal_limit = total_break * args.normal_ratio

    if total_break > 0 and total_normal < normal_limit:
        process_category(
            servers_data,
            output_dir,
            stats,
            category="normal",
            normal_limit=normal_limit,
            already_saved_normal=total_normal,
        )
        total_break, total_normal = count_total_saved(output_dir)

    summary_path = save_summary(
        output_dir=output_dir,
        source_json=input_json,
        stats=stats,
        normal_limit=normal_limit,
        total_break=total_break,
        total_normal=total_normal,
    )

    print("\n수집 완료")
    print(f"  break 전주: {total_break}")
    print(f"  normal 전주: {total_normal}")
    print(f"  summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
