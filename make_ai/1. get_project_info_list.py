#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""각 서버에서 프로젝트 목록을 조회하여 JSON 파일로 저장"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import poledb as PDB

# 서버 정보 (JT 서버는 데이터 수집 대상에서 제외)
SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}

OUTPUT_DIR = "1. project_info_list"
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


def _calc_summary(project_list):
    """프로젝트 목록에서 전체 통계 요약 계산"""
    total_poles = sum(p["statistics"]["total_poles"] for p in project_list)
    total_not_measured = sum(p["statistics"]["not_measured"] for p in project_list)
    total_measured = total_poles - total_not_measured
    total_anal1 = sum(p["statistics"]["anal1_completed"] for p in project_list)
    total_anal2 = sum(p["statistics"]["anal2_completed"] for p in project_list)
    total_anal1_break = sum(p["statistics"].get("anal1_break_count", 0) for p in project_list)
    total_anal2_break = sum(p["statistics"].get("anal2_break_count", 0) for p in project_list)

    anal2_ratio = (total_anal2 / total_poles * 100) if total_poles > 0 else 0
    anal1_break_ratio = (total_anal1_break / total_anal1 * 100) if total_anal1 > 0 else 0.0
    anal2_break_ratio = (total_anal2_break / total_anal2 * 100) if total_anal2 > 0 else 0.0

    return {
        "total_poles": total_poles,
        "total_not_measured": total_not_measured,
        "total_measured": total_measured,
        "total_anal1_completed": total_anal1,
        "total_anal2_completed": total_anal2,
        "overall_anal2_ratio": round(anal2_ratio, 2),
        "total_anal1_break": total_anal1_break,
        "total_anal2_break": total_anal2_break,
        "anal1_break_ratio": round(anal1_break_ratio, 2),
        "anal2_break_ratio": round(anal2_break_ratio, 2),
    }


def get_anal2_completed_count(project_name):
    """
    프로젝트에서 2차 분석 완료된 전주 수를 직접 조회
    
    Args:
        project_name: 프로젝트 이름
    
    Returns:
        int: 2차 분석 완료된 전주 수
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            return 0
        
        # 2차 분석 완료된 전주 조회
        # 1. tb_anal_state에서 anal2finyn이 NULL이 아닌 전주
        # 2. tb_anal_result에 analstep=2인 결과가 있는 전주
        query = """
            SELECT COUNT(DISTINCT tas.poleid) as count
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
            WHERE tas.groupname = %s
            AND tas.anal2finyn IS NOT NULL
            AND tar.analstep = 2
        """
        
        data = [project_name]
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        if result is None or result.empty:
            return 0
        
        return int(result.iloc[0]['count'])
        
    except Exception as e:
        print(f"    경고: 2차 분석 완료 전주 수 조회 실패: {e}")
        return 0

def get_anal_break_counts(project_name):
    """
    프로젝트에서 1차·2차 분석 완료 전주 중 파단(breakstate='B') 전주 수를 조회
    
    Returns:
        tuple: (anal1_break_count, anal2_break_count)
    """
    try:
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            return 0, 0
        # 1차 분석 완료 중 파단 (breakstate 'B')
        q1 = """
            SELECT COUNT(DISTINCT tas.poleid) as cnt
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid AND tar.analstep = 1 AND tar.breakstate = 'B'
            WHERE tas.groupname = %s
        """
        # 2차 분석 완료 중 파단
        q2 = """
            SELECT COUNT(DISTINCT tas.poleid) as cnt
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid AND tar.analstep = 2 AND tar.breakstate = 'B'
            WHERE tas.groupname = %s
        """
        r1 = PDB.poledb_conn.do_select_pd(q1, [project_name])
        r2 = PDB.poledb_conn.do_select_pd(q2, [project_name])
        c1 = int(r1.iloc[0]['cnt']) if r1 is not None and not r1.empty else 0
        c2 = int(r2.iloc[0]['cnt']) if r2 is not None and not r2.empty else 0
        return c1, c2
    except Exception as e:
        return 0, 0

def get_project_statistics(server, project_name):
    """프로젝트의 통계 정보 조회 (2차분석 여부와 무관하게 전체 수집)"""
    try:
        all_poles = PDB.get_pole_list_a(project_name)
        total_poles = len(all_poles) if all_poles is not None and not all_poles.empty else 0

        diag_progress = None
        if hasattr(PDB, 'group_diag_progress_info'):
            try:
                diag_progress = PDB.group_diag_progress_info(project_name)
            except Exception:
                diag_progress = None
        if diag_progress is None:
            diag_progress = {"total": total_poles, "-": 0, "MF": 0, "AP": 0, "AF": 0, "el": 0}

        anal_progress = None
        if hasattr(PDB, 'group_anal_progress_info'):
            try:
                anal_progress = PDB.group_anal_progress_info(project_name)
            except Exception:
                anal_progress = None
        if anal_progress is None:
            anal_progress = {"total": total_poles, "anal1": 0, "anal2": 0, "none": total_poles}

        anal2_count = get_anal2_completed_count(project_name)
        anal1_break, anal2_break = get_anal_break_counts(project_name)

        anal_progress["anal2"] = anal2_count
        anal1_count = anal_progress.get("anal1", 0)
        anal2_ratio = (anal2_count / total_poles * 100) if total_poles > 0 else 0
        anal1_break_ratio = (anal1_break / anal1_count * 100) if anal1_count > 0 else 0.0
        anal2_break_ratio = (anal2_break / anal2_count * 100) if anal2_count > 0 else 0.0
        
        return {
            "total_poles": total_poles,
            "not_measured": diag_progress.get("-", 0),  # 미측정
            "not_analyzed": anal_progress.get("none", 0),  # 미분석
            "anal1_completed": anal1_count,  # 1차 분석 완료
            "anal2_completed": anal_progress.get("anal2", 0),  # 2차 분석 완료
            "anal2_ratio": round(anal2_ratio, 2),  # 2차 분석 완료 비율 (%)
            "anal1_break_count": anal1_break,   # 1차 분석 완료 중 파단 전주 수
            "anal2_break_count": anal2_break,   # 2차 분석 완료 중 파단 전주 수
            "anal1_break_ratio": round(anal1_break_ratio, 2),  # 1차 완료 중 파단 비율 (%)
            "anal2_break_ratio": round(anal2_break_ratio, 2),  # 2차 완료 중 파단 비율 (%)
        }
    except Exception as e:
        print(f"  [{project_name}] 통계 조회 오류: {e}")
        return None

def get_project_list_from_server(server):
    """지정 서버에서 프로젝트 목록 및 통계 조회 (2차분석 여부와 무관하게 전체 수집)"""
    try:
        print(f"\n[{SERVERS[server]}] 연결 중...")
        PDB.poledb_init(server)

        project_list = PDB.groupname_info()
        if project_list is None:
            print(f"[{SERVERS[server]}] 프로젝트 목록 조회 실패")
            return []

        print(f"[{SERVERS[server]}] 전체 프로젝트 개수: {len(project_list)}개")
        print(f"[{SERVERS[server]}] 각 프로젝트의 통계 정보 조회 중...")

        projects_with_stats = []
        total_projects = len(project_list)
        pbar = tqdm(
            total=total_projects,
            desc=f"  [{SERVERS[server]}] 프로젝트 처리",
            unit="프로젝트",
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            ncols=100,
            dynamic_ncols=False
        )
        
        for project_name in project_list:
            stats = get_project_statistics(server, project_name)
            pbar.update(1)

            if stats is None:
                project_info = {"project_name": project_name, "statistics": DEFAULT_STATS.copy()}
            else:
                project_info = {"project_name": project_name, "statistics": stats}
            
            projects_with_stats.append(project_info)

        pbar.close()
        
        print(f"[{SERVERS[server]}] 통계 정보 조회 완료: {len(projects_with_stats)}개")
        return projects_with_stats
        
    except Exception as e:
        print(f"[{SERVERS[server]}] 오류 발생: {e}")
        return []

def save_project_list_to_json(server, project_list):
    """프로젝트 목록을 JSON 파일로 저장"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(current_dir) / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = output_dir / f"project_list_{server}_{timestamp}.json"
    summary = _calc_summary(project_list)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "server": server,
            "server_name": SERVERS[server],
            "timestamp": timestamp,
            "total_count": len(project_list),
            "summary": summary,
            "projects": project_list,
            "project_names": [p["project_name"] for p in project_list],
        }, f, ensure_ascii=False, indent=2)

    print(f"[{SERVERS[server]}] 저장 완료: {filename}")
    return filename

def save_all_servers_project_list():
    """모든 서버에서 프로젝트 목록을 조회하여 저장"""
    all_projects = {}
    summary = {}

    for server in tqdm(SERVERS.keys(), desc="서버 처리", unit="서버", leave=True):
        project_list = get_project_list_from_server(server)
        if not project_list:
            continue

        server_summary = _calc_summary(project_list)

        all_projects[server] = {
            "server_name": SERVERS[server],
            "projects": project_list,
            "project_names": [p["project_name"] for p in project_list],
            "count": len(project_list),
            "summary": server_summary,
        }
        summary[server] = len(project_list)

    # 전체 통합 파일 저장
    all_project_list = [p for projs in all_projects.values() for p in projs["projects"]]
    overall_summary = _calc_summary(all_project_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(current_dir) / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_filename = output_dir / f"project_list_all_{timestamp}.json"
    with open(all_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "servers": all_projects,
            "summary": {
                "by_server": summary,
                "total_projects": sum(summary.values()),
                **overall_summary,
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"\n전체 통합 파일 저장 완료: {all_filename}")
    _print_summary(all_projects, summary, overall_summary)
    return all_filename


def _print_summary(all_projects, summary, overall):
    """서버별/전체 요약 출력"""
    print("\n=== 서버별 프로젝트 개수 ===")
    for server, count in summary.items():
        s = all_projects[server]["summary"]
        print(f"{SERVERS[server]}: {count}개 프로젝트, 전주: {s['total_poles']}개, 미측정: {s['total_not_measured']}, 측정: {s['total_measured']}, "
              f"1차완료: {s['total_anal1_completed']}(파단 {s['total_anal1_break']}·{s['anal1_break_ratio']}%), "
              f"2차완료: {s['total_anal2_completed']}(파단 {s['total_anal2_break']}·{s['anal2_break_ratio']}%, 전체대비 {s['overall_anal2_ratio']}%)")
    print(f"전체: {sum(summary.values())}개 프로젝트 | 전주: {overall['total_poles']} | 미측정: {overall['total_not_measured']} | 측정: {overall['total_measured']} | "
          f"1차완료: {overall['total_anal1_completed']}(파단 {overall['total_anal1_break']}·{overall['anal1_break_ratio']}%) | "
          f"2차완료: {overall['total_anal2_completed']}(파단 {overall['total_anal2_break']}·{overall['anal2_break_ratio']}%, 전체대비 {overall['overall_anal2_ratio']}%)")

if __name__ == "__main__":
    print("=" * 60)
    print("프로젝트 목록 조회 및 저장 시작")
    print("=" * 60)
    save_all_servers_project_list()
    
    print("\n" + "=" * 60)
    print("프로젝트 목록 조회 및 저장 완료")
    print("=" * 60)

