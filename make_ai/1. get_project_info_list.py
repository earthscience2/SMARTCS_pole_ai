#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
각 서버에서 프로젝트 목록을 조회하여 JSON 파일로 저장하는 스크립트
"""

import sys
import os
import json
from datetime import datetime

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

def get_project_statistics(server, project_name):
    """
    프로젝트의 통계 정보를 조회
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
    
    Returns:
        dict: 프로젝트 통계 정보
    """
    try:
        # 전체 전주 목록 조회
        all_poles = PDB.get_pole_list_a(project_name)
        total_poles = len(all_poles) if all_poles is not None and not all_poles.empty else 0
        
        if total_poles == 0:
            return None
        
        # 측정 진행도 조회 (함수가 없을 수 있으므로 try-except 사용)
        diag_progress = None
        if hasattr(PDB, 'group_diag_progress_info'):
            try:
                diag_progress = PDB.group_diag_progress_info(project_name)
            except Exception as e:
                print(f"    경고: group_diag_progress_info 호출 실패: {e}")
                diag_progress = None
        
        if diag_progress is None:
            diag_progress = {"total": total_poles, "-": 0, "MF": 0, "AP": 0, "AF": 0, "el": 0}
        
        # 분석 진행도 조회 (함수가 없을 수 있으므로 try-except 사용)
        anal_progress = None
        if hasattr(PDB, 'group_anal_progress_info'):
            try:
                anal_progress = PDB.group_anal_progress_info(project_name)
            except Exception as e:
                print(f"    경고: group_anal_progress_info 호출 실패: {e}")
                anal_progress = None
        
        if anal_progress is None:
            anal_progress = {"total": total_poles, "anal1": 0, "anal2": 0, "none": total_poles}
        
        # 2차 분석 완료 전주 수 직접 조회 (DB 쿼리로)
        anal2_count = get_anal2_completed_count(project_name)
        
        # 기존 anal_progress에 실제 조회한 값으로 업데이트
        anal_progress["anal2"] = anal2_count
        
        # 2차 분석 완료 비율 계산
        anal2_ratio = (anal2_count / total_poles * 100) if total_poles > 0 else 0
        
        return {
            "total_poles": total_poles,
            "not_measured": diag_progress.get("-", 0),  # 미측정
            "not_analyzed": anal_progress.get("none", 0),  # 미분석
            "anal1_completed": anal_progress.get("anal1", 0),  # 1차 분석 완료
            "anal2_completed": anal_progress.get("anal2", 0),  # 2차 분석 완료
            "anal2_ratio": round(anal2_ratio, 2)  # 2차 분석 완료 비율 (%)
        }
    except Exception as e:
        print(f"  [{project_name}] 통계 조회 오류: {e}")
        return None

def get_project_list_from_server(server):
    """
    지정된 서버에서 프로젝트 목록을 조회하고 각 프로젝트의 통계 정보를 포함
    
    Args:
        server: 서버 이름 (main, is, kh, jt)
    
    Returns:
        list: 프로젝트 목록 (통계 정보 포함)
    """
    try:
        print(f"\n[{SERVERS[server]}] 연결 중...")
        PDB.poledb_init(server)
        
        # 프로젝트 목록 조회
        project_list = PDB.groupname_info()
        
        if project_list is None:
            print(f"[{SERVERS[server]}] 프로젝트 목록 조회 실패")
            return []
        
        print(f"[{SERVERS[server]}] 전체 프로젝트 개수: {len(project_list)}개")
        print(f"[{SERVERS[server]}] 각 프로젝트의 통계 정보 조회 중...")
        
        projects_with_stats = []
        for i, project_name in enumerate(project_list, 1):
            print(f"  [{i}/{len(project_list)}] {project_name} 통계 조회 중...", end=" ")
            
            # 프로젝트 통계 정보 조회
            stats = get_project_statistics(server, project_name)
            
            if stats is None:
                print("통계 정보 없음")
                # 통계 정보가 없어도 프로젝트는 포함
                project_info = {
                    "project_name": project_name,
                    "statistics": {
                        "total_poles": 0,
                        "not_measured": 0,
                        "not_analyzed": 0,
                        "anal1_completed": 0,
                        "anal2_completed": 0,
                        "anal2_ratio": 0.0
                    }
                }
            else:
                project_info = {
                    "project_name": project_name,
                    "statistics": stats
                }
                print(f"완료 (전체: {stats['total_poles']}개, 2차분석: {stats['anal2_ratio']}%)")
            
            projects_with_stats.append(project_info)
        
        print(f"[{SERVERS[server]}] 통계 정보 조회 완료: {len(projects_with_stats)}개")
        return projects_with_stats
        
    except Exception as e:
        print(f"[{SERVERS[server]}] 오류 발생: {e}")
        return []

def save_project_list_to_json(server, project_list):
    """
    프로젝트 목록을 JSON 파일로 저장
    
    Args:
        server: 서버 이름
        project_list: 프로젝트 목록 (통계 정보 포함)
    """
    # 출력 디렉토리 생성 (make_ai 폴더 내부)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "1. project_info_list")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/project_list_{server}_{timestamp}.json"
    
    # 프로젝트 이름만 추출 (통계 정보는 유지)
    project_names = [p["project_name"] for p in project_list]
    
    # 통계 요약 계산
    total_poles_sum = sum(p["statistics"]["total_poles"] for p in project_list)
    total_anal2_sum = sum(p["statistics"]["anal2_completed"] for p in project_list)
    overall_anal2_ratio = (total_anal2_sum / total_poles_sum * 100) if total_poles_sum > 0 else 0
    
    # JSON 파일로 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "server": server,
            "server_name": SERVERS[server],
            "timestamp": timestamp,
            "total_count": len(project_list),
            "summary": {
                "total_poles": total_poles_sum,
                "total_anal2_completed": total_anal2_sum,
                "overall_anal2_ratio": round(overall_anal2_ratio, 2)
            },
            "projects": project_list,
            "project_names": project_names  # 간단한 이름 리스트도 포함
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[{SERVERS[server]}] 저장 완료: {filename}")
    return filename

def save_all_servers_project_list():
    """
    모든 서버에서 프로젝트 목록을 조회하여 저장 (전체 프로젝트, 통계 정보 포함)
    """
    all_projects = {}
    summary = {}
    
    for server in SERVERS.keys():
        project_list = get_project_list_from_server(server)
        
        if project_list:
            # 개별 서버 파일 저장
            save_project_list_to_json(server, project_list)
            
            # 전체 통합 데이터에 추가
            project_names = [p["project_name"] for p in project_list]
            
            # 서버별 통계 요약 계산
            total_poles_sum = sum(p["statistics"]["total_poles"] for p in project_list)
            total_anal2_sum = sum(p["statistics"]["anal2_completed"] for p in project_list)
            overall_anal2_ratio = (total_anal2_sum / total_poles_sum * 100) if total_poles_sum > 0 else 0
            
            all_projects[server] = {
                "server_name": SERVERS[server],
                "projects": project_list,  # 통계 정보 포함
                "project_names": project_names,  # 이름만 리스트
                "count": len(project_list),
                "summary": {
                    "total_poles": total_poles_sum,
                    "total_anal2_completed": total_anal2_sum,
                    "overall_anal2_ratio": round(overall_anal2_ratio, 2)
                }
            }
            
            summary[server] = len(project_list)
    
    # 전체 통합 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "1. project_info_list")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 전체 통합 통계 계산
    all_total_poles = sum(s["summary"]["total_poles"] for s in all_projects.values())
    all_total_anal2 = sum(s["summary"]["total_anal2_completed"] for s in all_projects.values())
    all_overall_ratio = (all_total_anal2 / all_total_poles * 100) if all_total_poles > 0 else 0
    
    all_filename = f"{output_dir}/project_list_all_{timestamp}.json"
    with open(all_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "servers": all_projects,
            "summary": {
                "by_server": summary,
                "total_projects": sum(summary.values()),
                "total_poles": all_total_poles,
                "total_anal2_completed": all_total_anal2,
                "overall_anal2_ratio": round(all_overall_ratio, 2)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n전체 통합 파일 저장 완료: {all_filename}")
    
    # 요약 출력
    print("\n=== 서버별 프로젝트 개수 ===")
    for server, count in summary.items():
        server_summary = all_projects[server]["summary"]
        print(f"{SERVERS[server]}: {count}개 프로젝트, 전체 전주: {server_summary['total_poles']}개, 2차분석 완료: {server_summary['overall_anal2_ratio']}%")
    print(f"전체: {sum(summary.values())}개 프로젝트, 전체 전주: {all_total_poles}개, 2차분석 완료: {round(all_overall_ratio, 2)}%")
    
    return all_filename

if __name__ == "__main__":
    print("=" * 50)
    print("프로젝트 목록 조회 및 저장 시작")
    print("전체 프로젝트 조회 (통계 정보 포함)")
    print("=" * 50)
    
    # 모든 서버에서 프로젝트 목록 조회 및 저장
    save_all_servers_project_list()
    
    print("\n" + "=" * 50)
    print("프로젝트 목록 조회 및 저장 완료")
    print("=" * 50)

