#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""프로젝트별 분석 완료 전주 목록(최신 분석단계 기준)을 JSON으로 저장한다."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import poledb as PDB


SERVERS: Dict[str, str] = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}
PROJECT_INFO_DIR = CURRENT_DIR / "1. project_info_list"
OUTPUT_DIR = CURRENT_DIR / "2. anal_pole_list"


def find_latest_project_list_json(input_dir: Path) -> Path:
    """`project_list_all_*.json` 중 최신 파일을 찾는다."""
    candidates = glob.glob(str(input_dir / "project_list_all_*.json"))
    if not candidates:
        raise FileNotFoundError(f"project_list_all_*.json 파일이 없습니다: {input_dir}")
    return Path(max(candidates, key=lambda p: Path(p).stat().st_mtime))


def load_target_projects(json_path: Path, server: str) -> List[str]:
    """통합 JSON에서 1차 이상 분석된 프로젝트 이름 목록을 읽는다."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    server_data = payload.get("servers", {}).get(server)
    if not server_data:
        return []

    projects = server_data.get("projects", [])
    out: List[str] = []
    for project in projects:
        stats = project.get("statistics", {})
        if stats.get("anal1_completed", 0) > 0 or stats.get("anal2_completed", 0) > 0:
            name = project.get("project_name")
            if name:
                out.append(name)
    return out


def query_project_poles(project_name: str) -> List[Dict[str, str]]:
    """프로젝트 내 전주별 최신(2차 우선) 분석 결과를 조회한다."""
    if not hasattr(PDB, "poledb_conn") or PDB.poledb_conn is None:
        return []

    query = """
        SELECT
            tas.poleid,
            COALESCE(tar.breakstate, 'N') AS breakstate
        FROM tb_anal_state tas
        INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
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
        GROUP BY tas.poleid, tar.breakstate
        ORDER BY tas.poleid
    """
    df = PDB.poledb_conn.do_select_pd(query, [project_name])
    if df is None or df.empty:
        return []

    poles: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        poleid = str(row["poleid"]).strip()
        breakstate = str(row["breakstate"]).strip().upper() if pd.notna(row["breakstate"]) else "N"
        poles.append({"poleid": poleid, "breakstate": breakstate})
    return poles


def collect_server_data(server: str, projects: List[str]) -> Dict[str, Dict]:
    """단일 서버의 프로젝트별 전주 목록을 수집한다."""
    print(f"\n[{SERVERS[server]}] DB 연결")
    PDB.poledb_init(server)

    project_map: Dict[str, Dict] = {}
    total_break = 0
    total_normal = 0
    for project_name in tqdm(projects, desc=f"[{SERVERS[server]}] 프로젝트", unit="project"):
        poles = query_project_poles(project_name)
        break_count = sum(1 for p in poles if p["breakstate"] == "B")
        normal_count = len(poles) - break_count
        total_break += break_count
        total_normal += normal_count
        project_map[project_name] = {
            "pole_count": len(poles),
            "break_count": break_count,
            "normal_count": normal_count,
            "pole_ids": [p["poleid"] for p in poles],
            "poles_info": {p["poleid"]: p["breakstate"] for p in poles},
        }

    print(
        f"[{SERVERS[server]}] 완료: 전주 {total_break + total_normal}개 "
        f"(파단 {total_break}, 정상 {total_normal})"
    )
    return project_map


def save_server_json(server: str, project_map: Dict[str, Dict], output_dir: Path, timestamp: str) -> Path:
    """서버 단위 JSON을 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_poles = sum(v["pole_count"] for v in project_map.values())
    out_path = output_dir / f"anal_poles_{server}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "server": server,
                "server_name": SERVERS[server],
                "timestamp": timestamp,
                "total_projects": len(project_map),
                "total_anal_poles": total_poles,
                "projects": project_map,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def save_all_json(all_data: Dict[str, Dict], output_dir: Path, timestamp: str) -> Path:
    """전체 서버 통합 JSON을 저장한다."""
    summary = {
        server: {"projects": payload["total_projects"], "poles": payload["total_anal_poles"]}
        for server, payload in all_data.items()
    }
    out_path = output_dir / f"anal2_poles_all_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "servers": all_data,
                "summary": {
                    "by_server": summary,
                    "total_projects": sum(x["projects"] for x in summary.values()),
                    "total_anal_poles": sum(x["poles"] for x in summary.values()),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def run(server: str, project_json: Path, output_dir: Path) -> Optional[Path]:
    """전체 실행 함수."""
    target_servers = [server] if server != "all" else list(SERVERS.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_data: Dict[str, Dict] = {}

    for sv in target_servers:
        projects = load_target_projects(project_json, sv)
        if not projects:
            print(f"[{SERVERS[sv]}] 대상 프로젝트 없음")
            continue
        project_map = collect_server_data(sv, projects)
        save_server_json(sv, project_map, output_dir, timestamp)
        all_data[sv] = {
            "server_name": SERVERS[sv],
            "projects": project_map,
            "total_projects": len(project_map),
            "total_anal_poles": sum(v["pole_count"] for v in project_map.values()),
        }

    if not all_data:
        print("저장할 결과가 없습니다.")
        return None

    out_path = save_all_json(all_data, output_dir, timestamp)
    print(f"\n통합 JSON 저장: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="분석 완료 전주 목록 수집")
    parser.add_argument(
        "--server",
        default="all",
        choices=["all", *SERVERS.keys()],
        help="수집 대상 서버",
    )
    parser.add_argument(
        "--project-list-json",
        default=None,
        help="입력 project_list_all_*.json 경로 (미지정 시 최신 파일 자동 선택)",
    )
    parser.add_argument(
        "--project-list-dir",
        default=str(PROJECT_INFO_DIR),
        help="project_list_all_*.json 검색 디렉터리",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="출력 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.project_list_json:
        project_json = Path(args.project_list_json)
    else:
        project_json = find_latest_project_list_json(Path(args.project_list_dir))
    if not project_json.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {project_json}")

    print("=" * 70)
    print("2단계: 분석 완료 전주 목록 수집 시작")
    print(f"입력: {project_json}")
    print("=" * 70)
    run(args.server, project_json, output_dir)
    print("\n" + "=" * 70)
    print("2단계: 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
