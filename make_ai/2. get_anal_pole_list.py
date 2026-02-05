#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""각 프로젝트별로 1차 분석 이상 완료된 전주 ID를 조회하여 JSON 파일로 저장"""

import sys
import os
import json
import glob
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
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

PROJECT_INFO_DIR = "1. project_info_list"
OUTPUT_DIR = "2. anal_pole_list"


def find_latest_project_list_json():
    """1. project_info_list 폴더에서 가장 최신 통합 프로젝트 목록 JSON 경로 반환 (project_list_all_*.json)"""
    json_dir = os.path.join(current_dir, PROJECT_INFO_DIR)
    pattern_all = os.path.join(json_dir, "project_list_all_*.json")
    json_files = list(glob.glob(pattern_all))

    if not json_files:
        raise FileNotFoundError(f"프로젝트 목록 JSON을 찾을 수 없습니다: {pattern_all}")

    latest_file = max(json_files, key=os.path.getmtime)
    print(f"최신 프로젝트 목록 JSON 파일: {latest_file}")
    return latest_file

def get_project_list_from_json(json_file_path, server):
    """통합 JSON(project_list_all_*.json)에서 해당 서버의 1차 분석 이상 완료 프로젝트 이름 목록 추출"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'servers' not in data:
        print(f"    경고: 통합 형식(servers)이 아닙니다. (파일: {os.path.basename(json_file_path)})")
        return []

    server_data = data.get('servers', {}).get(server)
    if not server_data:
        print(f"    경고: {SERVERS[server]}의 데이터를 찾을 수 없습니다. (파일: {os.path.basename(json_file_path)})")
        return []

    projects = server_data.get('projects', [])
    return [
        p['project_name'] for p in projects
        if p.get('project_name')
        and (p.get('statistics', {}).get('anal1_completed', 0) > 0
             or p.get('statistics', {}).get('anal2_completed', 0) > 0)
    ]

def get_anal2_completed_poles(server, project_name):
    """1차 분석 이상 완료된 전주 목록 조회 (breakstate 포함, analstep 우선·regdate 최신 기준)"""
    try:
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{project_name}]: 데이터베이스 연결이 없습니다.")
            return []

        query = """
            SELECT 
                tas.poleid,
                COALESCE(tar.breakstate, 'N') as breakstate
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
            INNER JOIN (
                SELECT 
                    tar1.poleid,
                    MAX(tar1.analstep) as max_analstep
                FROM tb_anal_result tar1
                WHERE tar1.analstep IN (1, 2)
                GROUP BY tar1.poleid
            ) tar_max ON tar.poleid = tar_max.poleid
            INNER JOIN (
                SELECT 
                    tar2.poleid,
                    tar2.analstep,
                    MAX(tar2.regdate) as max_regdate
                FROM tb_anal_result tar2
                WHERE tar2.analstep IN (1, 2)
                GROUP BY tar2.poleid, tar2.analstep
            ) tar_latest ON tar.poleid = tar_latest.poleid 
                AND tar.analstep = tar_latest.analstep
                AND tar.regdate = tar_latest.max_regdate
                AND tar.analstep = tar_max.max_analstep
            WHERE tas.groupname = %s
            GROUP BY tas.poleid, tar.breakstate
            ORDER BY tas.poleid
        """
        
        result = PDB.poledb_conn.do_select_pd(query, [project_name])
        if result is None or result.empty:
            return []

        pole_list = []
        for _, row in result.iterrows():
            poleid = str(row['poleid']).strip()
            breakstate = str(row['breakstate']).strip().upper() if pd.notna(row['breakstate']) else 'N'
            pole_list.append({
                "poleid": poleid,
                "breakstate": breakstate
            })
        
        return pole_list
        
    except Exception as e:
        print(f"    [{project_name}] 분석 완료 전주 조회 오류: {e}")
        traceback.print_exc()
        return []

def get_project_anal2_poles(server, project_list):
    """서버의 모든 프로젝트에서 1차 분석 이상 완료 전주 정보 조회 (breakstate 포함)"""
    print(f"\n[{SERVERS[server]}] 연결 중...")
    PDB.poledb_init(server)

    print(f"[{SERVERS[server]}] 전체 프로젝트 개수: {len(project_list)}개")
    print(f"[{SERVERS[server]}] 각 프로젝트의 분석 완료 전주 조회 중...")

    project_poles = {}
    break_total, normal_total = 0, 0

    for project_name in tqdm(project_list, desc=f"  [{SERVERS[server]}] 프로젝트 처리", unit="프로젝트"):
        anal2_poles = get_anal2_completed_poles(server, project_name)
        break_count = sum(1 for p in anal2_poles if p.get('breakstate') == 'B')
        normal_count = len(anal2_poles) - break_count
        break_total += break_count
        normal_total += normal_count

        project_poles[project_name] = {
            "pole_count": len(anal2_poles),
            "break_count": break_count,
            "normal_count": normal_count,
            "pole_ids": [p['poleid'] for p in anal2_poles],
            "poles_info": {p['poleid']: p['breakstate'] for p in anal2_poles},
        }

    print(f"[{SERVERS[server]}] 조회 완료 (전체: {break_total + normal_total}개, 파단: {break_total}개, 정상: {normal_total}개)")
    return project_poles

def save_anal2_poles_to_json(server, project_poles):
    """1차 분석 이상 완료 전주 목록을 JSON 파일로 저장"""
    output_dir = Path(current_dir) / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = output_dir / f"anal_poles_{server}_{timestamp}.json"
    total_poles = sum(info["pole_count"] for info in project_poles.values())

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "server": server,
            "server_name": SERVERS[server],
            "timestamp": timestamp,
            "total_projects": len(project_poles),
            "total_anal_poles": total_poles,
            "projects": project_poles,
        }, f, ensure_ascii=False, indent=2)

    print(f"[{SERVERS[server]}] 저장 완료: {filename}")
    return filename

def get_all_servers_anal2_poles():
    """모든 서버에서 1차 분석 이상 완료 전주 ID 조회 및 저장"""
    all_servers_data = {}
    summary = {}
    
    try:
        project_list_json_file = find_latest_project_list_json()
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print(f"{PROJECT_INFO_DIR} 폴더에 project_list_all_*.json 파일이 필요합니다.")
        return None
    
    for server in SERVERS.keys():
        try:
            print(f"\n[{SERVERS[server]}] 프로젝트 목록 읽는 중...")
            project_list = get_project_list_from_json(project_list_json_file, server)

            if not project_list:
                print(f"[{SERVERS[server]}] 분석 완료된 프로젝트가 없습니다.")
                continue

            print(f"[{SERVERS[server]}] 분석 완료 프로젝트: {len(project_list)}개")
            project_poles = get_project_anal2_poles(server, project_list)

            if project_poles:
                total_poles = sum(info["pole_count"] for info in project_poles.values())
                all_servers_data[server] = {
                    "server_name": SERVERS[server],
                    "projects": project_poles,
                    "total_projects": len(project_poles),
                    "total_anal_poles": total_poles,
                }
                
                summary[server] = {
                    "projects": len(project_poles),
                    "poles": total_poles
                }
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 오류 발생: {e}")
            continue
    
    if all_servers_data:
        output_dir = Path(current_dir) / OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        all_total_projects = sum(s["total_projects"] for s in all_servers_data.values())
        all_total_poles = sum(s["total_anal_poles"] for s in all_servers_data.values())
        
        all_filename = output_dir / f"anal2_poles_all_{timestamp}.json"
        with open(all_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "servers": all_servers_data,
                "summary": {
                    "by_server": summary,
                    "total_projects": all_total_projects,
                    "total_anal_poles": all_total_poles,
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n전체 통합 파일 저장 완료: {all_filename}")
        _print_summary(summary)
        return all_filename
    
    return None


def _print_summary(summary):
    """서버별 요약 출력"""
    print("\n=== 서버별 분석 완료 전주 통계 (1차 분석 이상) ===")
    for server, info in summary.items():
        print(f"{SERVERS[server]}: {info['projects']}개 프로젝트, {info['poles']}개 전주")
    total_projects = sum(s["projects"] for s in summary.values())
    total_poles = sum(s["poles"] for s in summary.values())
    print(f"전체: {total_projects}개 프로젝트, {total_poles}개 전주")


if __name__ == "__main__":
    print("=" * 60)
    print("분석 완료 전주 ID 조회 및 저장 시작 (1차 분석 이상)")
    print("=" * 60)
    get_all_servers_anal2_poles()
    
    print("\n" + "=" * 60)
    print("분석 완료 전주 ID 조회 및 저장 완료 (1차 분석 이상)")
    print("=" * 60)

