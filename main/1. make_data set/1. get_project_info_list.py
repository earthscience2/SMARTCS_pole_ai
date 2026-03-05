#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""서버별 프로젝트 목록과 분석 진행 통계를 조회해 JSON으로 저장한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent  # main의 상위의 상위 (프로젝트 루트)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import poledb as PDB


SERVERS: Dict[str, str] = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "1. project_info_list"
DEFAULT_STATS = {
    "total_poles": 0,
    "not_measured": 0,
    "not_analyzed": 0,
    "anal1_completed": 0,
    "anal2_completed": 0,
    "anal2_ratio": 0.0,
    "anal1_break_count": 0,
    "anal2_break_count": 0,
    "anal1_break_ratio": 0.0,
    "anal2_break_ratio": 0.0,
}


def calc_summary(projects: List[Dict]) -> Dict:
    """프로젝트 목록에서 합계 통계를 계산한다."""
    total_poles = sum(p["statistics"]["total_poles"] for p in projects)
    total_not_measured = sum(p["statistics"]["not_measured"] for p in projects)
    total_measured = total_poles - total_not_measured
    total_anal1 = sum(p["statistics"]["anal1_completed"] for p in projects)
    total_anal2 = sum(p["statistics"]["anal2_completed"] for p in projects)
    total_anal1_break = sum(p["statistics"].get("anal1_break_count", 0) for p in projects)
    total_anal2_break = sum(p["statistics"].get("anal2_break_count", 0) for p in projects)

    return {
        "total_poles": total_poles,
        "total_not_measured": total_not_measured,
        "total_measured": total_measured,
        "total_anal1_completed": total_anal1,
        "total_anal2_completed": total_anal2,
        "overall_anal2_ratio": round((total_anal2 / total_poles * 100) if total_poles else 0.0, 2),
        "total_anal1_break": total_anal1_break,
        "total_anal2_break": total_anal2_break,
        "anal1_break_ratio": round((total_anal1_break / total_anal1 * 100) if total_anal1 else 0.0, 2),
        "anal2_break_ratio": round((total_anal2_break / total_anal2 * 100) if total_anal2 else 0.0, 2),
    }


def query_anal2_completed_count(project_name: str) -> int:
    """해당 프로젝트에서 2차 분석 완료 전주 수를 조회한다."""
    if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
        return 0

    query = """
        SELECT COUNT(DISTINCT tas.poleid) AS count
        FROM tb_anal_state tas
        INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
        WHERE tas.groupname = %s
          AND tas.anal2finyn IS NOT NULL
          AND tar.analstep = 2
    """
    result = PDB.poledb_conn.do_select_pd(query, [project_name])
    if result is None or result.empty:
        return 0
    return int(result.iloc[0]["count"])


def query_break_counts(project_name: str) -> tuple[int, int]:
    """1차/2차 분석 결과에서 파단(B) 전주 수를 조회한다."""
    if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
        return 0, 0

    q1 = """
        SELECT COUNT(DISTINCT tas.poleid) AS cnt
        FROM tb_anal_state tas
        INNER JOIN tb_anal_result tar
            ON tas.poleid = tar.poleid
           AND tar.analstep = 1
           AND tar.breakstate = 'B'
        WHERE tas.groupname = %s
    """
    q2 = """
        SELECT COUNT(DISTINCT tas.poleid) AS cnt
        FROM tb_anal_state tas
        INNER JOIN tb_anal_result tar
            ON tas.poleid = tar.poleid
           AND tar.analstep = 2
           AND tar.breakstate = 'B'
        WHERE tas.groupname = %s
    """
    r1 = PDB.poledb_conn.do_select_pd(q1, [project_name])
    r2 = PDB.poledb_conn.do_select_pd(q2, [project_name])
    c1 = int(r1.iloc[0]["cnt"]) if r1 is not None and not r1.empty else 0
    c2 = int(r2.iloc[0]["cnt"]) if r2 is not None and not r2.empty else 0
    return c1, c2


def get_project_statistics(project_name: str) -> Optional[Dict]:
    """단일 프로젝트의 통계 정보를 조회한다."""
    try:
        all_poles = PDB.get_pole_list_a(project_name)
        total_poles = len(all_poles) if all_poles is not None and not all_poles.empty else 0

        diag = None
        if hasattr(PDB, "group_diag_progress_info"):
            try:
                diag = PDB.group_diag_progress_info(project_name)
            except Exception:
                diag = None
        if diag is None:
            diag = {"total": total_poles, "-": 0, "MF": 0, "AP": 0, "AF": 0, "el": 0}

        anal = None
        if hasattr(PDB, "group_anal_progress_info"):
            try:
                anal = PDB.group_anal_progress_info(project_name)
            except Exception:
                anal = None
        if anal is None:
            anal = {"total": total_poles, "anal1": 0, "anal2": 0, "none": total_poles}

        anal2_count = query_anal2_completed_count(project_name)
        anal1_break, anal2_break = query_break_counts(project_name)

        anal1_count = int(anal.get("anal1", 0))
        anal2_ratio = (anal2_count / total_poles * 100) if total_poles else 0.0
        anal1_break_ratio = (anal1_break / anal1_count * 100) if anal1_count else 0.0
        anal2_break_ratio = (anal2_break / anal2_count * 100) if anal2_count else 0.0

        return {
            "total_poles": total_poles,
            "not_measured": int(diag.get("-", 0)),
            "not_analyzed": int(anal.get("none", 0)),
            "anal1_completed": anal1_count,
            "anal2_completed": anal2_count,
            "anal2_ratio": round(anal2_ratio, 2),
            "anal1_break_count": anal1_break,
            "anal2_break_count": anal2_break,
            "anal1_break_ratio": round(anal1_break_ratio, 2),
            "anal2_break_ratio": round(anal2_break_ratio, 2),
        }
    except Exception as exc:
        print(f"  [{project_name}] 통계 조회 실패: {exc}")
        return None


def get_projects_from_server(server: str) -> List[Dict]:
    """서버에서 프로젝트 목록과 통계를 수집한다."""
    print(f"\n[{SERVERS[server]}] DB 연결")
    PDB.poledb_init(server)

    project_list = PDB.groupname_info()
    if project_list is None:
        print(f"[{SERVERS[server]}] 프로젝트 목록 조회 실패")
        return []

    projects: List[Dict] = []
    pbar = tqdm(project_list, desc=f"[{SERVERS[server]}] 프로젝트", unit="project")
    for project_name in pbar:
        stats = get_project_statistics(project_name)
        projects.append({
            "project_name": project_name,
            "statistics": stats if stats is not None else DEFAULT_STATS.copy(),
        })
    print(f"[{SERVERS[server]}] 수집 완료: {len(projects)}개")
    return projects


def save_server_json(server: str, project_list: List[Dict], output_dir: Path, timestamp: str) -> Path:
    """서버 단위 결과 JSON을 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"project_list_{server}_{timestamp}.json"
    payload = {
        "server": server,
        "server_name": SERVERS[server],
        "timestamp": timestamp,
        "total_count": len(project_list),
        "summary": calc_summary(project_list),
        "projects": project_list,
        "project_names": [p["project_name"] for p in project_list],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def save_all_servers(output_dir: Path, only_server: str = "all") -> Optional[Path]:
    """전체 또는 단일 서버 결과를 저장한다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    target_servers = [only_server] if only_server != "all" else list(SERVERS.keys())

    all_servers: Dict[str, Dict] = {}
    summary_by_server: Dict[str, int] = {}
    merged_projects: List[Dict] = []

    for server in target_servers:
        projects = get_projects_from_server(server)
        if not projects:
            continue
        save_server_json(server, projects, output_dir, timestamp)

        server_summary = calc_summary(projects)
        all_servers[server] = {
            "server_name": SERVERS[server],
            "projects": projects,
            "project_names": [p["project_name"] for p in projects],
            "count": len(projects),
            "summary": server_summary,
        }
        summary_by_server[server] = len(projects)
        merged_projects.extend(projects)

    if not all_servers:
        print("저장할 데이터가 없습니다.")
        return None

    overall_summary = calc_summary(merged_projects)
    out_path = output_dir / f"project_list_all_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "servers": all_servers,
                "summary": {
                    "by_server": summary_by_server,
                    "total_projects": sum(summary_by_server.values()),
                    **overall_summary,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(all_servers, summary_by_server, overall_summary)
    print(f"\n통합 JSON 저장: {out_path}")
    return out_path


def print_summary(all_servers: Dict[str, Dict], summary: Dict[str, int], overall: Dict) -> None:
    """서버별/전체 요약을 출력한다."""
    print("\n=== 서버별 요약 ===")
    for server, count in summary.items():
        s = all_servers[server]["summary"]
        print(
            f"{SERVERS[server]}: 프로젝트 {count}개, 전주 {s['total_poles']}개, "
            f"1차 {s['total_anal1_completed']}개(파단 {s['total_anal1_break']}개), "
            f"2차 {s['total_anal2_completed']}개(파단 {s['total_anal2_break']}개)"
        )
    print(
        f"전체: 프로젝트 {sum(summary.values())}개, 전주 {overall['total_poles']}개, "
        f"1차 {overall['total_anal1_completed']}개, 2차 {overall['total_anal2_completed']}개"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="프로젝트 목록/통계 조회")
    parser.add_argument(
        "--server",
        default="all",
        choices=["all", *SERVERS.keys()],
        help="조회 대상 서버 (기본: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="결과 저장 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    print("=" * 70)
    print("1단계: 프로젝트 목록/통계 수집 시작")
    print("=" * 70)
    save_all_servers(output_dir=output_dir, only_server=args.server)
    print("\n" + "=" * 70)
    print("1단계: 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
