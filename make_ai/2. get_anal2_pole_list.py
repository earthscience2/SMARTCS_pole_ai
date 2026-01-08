#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
각 프로젝트별로 2차 분석까지 완료된 전주 ID를 조회하여 정리하는 스크립트
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

import poledb as PDB

# 서버 정보
SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
    "jt": "제이티엔지니어링"
}

def get_anal2_completed_poles(server, project_name):
    """
    프로젝트에서 2차 분석까지 완료된 전주 ID 목록을 조회
    - anal2finyn이 NULL이 아닌 경우 (2차 분석 완료 표시)
    - tb_anal_result에 analstep=2인 결과가 있는지 확인
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
    
    Returns:
        list: 2차 분석 완료된 전주 ID 목록
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{project_name}]: 데이터베이스 연결이 없습니다.")
            return []
        
        # 2차 분석 완료된 전주 조회
        # 1. tb_anal_state에서 anal2finyn이 NULL이 아닌 전주
        # 2. tb_anal_result에 analstep=2인 결과가 있는 전주
        # 두 조건을 모두 만족하는 전주만 반환
        query = """
            SELECT DISTINCT tas.poleid
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
            WHERE tas.groupname = %s
            AND tas.anal2finyn IS NOT NULL
            AND tar.analstep = 2
            ORDER BY tas.poleid
        """
        
        data = [project_name]
        
        # poledb_conn에 직접 접근
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        if result is None or result.empty:
            return []
        
        pole_list = result['poleid'].tolist()
        return pole_list
        
    except Exception as e:
        print(f"    [{project_name}] 2차 분석 완료 전주 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_project_anal2_poles(server, project_list):
    """
    서버의 모든 프로젝트에서 2차 분석 완료된 전주 ID를 조회
    
    Args:
        server: 서버 이름
        project_list: 프로젝트 목록
    
    Returns:
        dict: 프로젝트별 2차 분석 완료 전주 ID 목록
    """
    print(f"\n[{SERVERS[server]}] 연결 중...")
    PDB.poledb_init(server)
    
    project_poles = {}
    
    print(f"[{SERVERS[server]}] 전체 프로젝트 개수: {len(project_list)}개")
    print(f"[{SERVERS[server]}] 각 프로젝트의 2차 분석 완료 전주 조회 중...")
    
    for i, project_name in enumerate(project_list, 1):
        print(f"  [{i}/{len(project_list)}] {project_name} 조회 중...", end=" ")
        
        # 2차 분석 완료된 전주 ID 조회
        anal2_poles = get_anal2_completed_poles(server, project_name)
        
        project_poles[project_name] = {
            "pole_count": len(anal2_poles),
            "pole_ids": anal2_poles
        }
        
        print(f"완료 ({len(anal2_poles)}개)")
    
    print(f"[{SERVERS[server]}] 조회 완료")
    return project_poles

def save_anal2_poles_to_json(server, project_poles):
    """
    2차 분석 완료 전주 목록을 JSON 파일로 저장
    
    Args:
        server: 서버 이름
        project_poles: 프로젝트별 전주 ID 목록
    """
    # 출력 디렉토리 생성 (make_ai 폴더 내부)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "2. anal2_pole_list")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/anal2_poles_{server}_{timestamp}.json"
    
    # 전체 전주 수 계산
    total_poles = sum(info["pole_count"] for info in project_poles.values())
    
    # JSON 파일로 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "server": server,
            "server_name": SERVERS[server],
            "timestamp": timestamp,
            "total_projects": len(project_poles),
            "total_anal2_poles": total_poles,
            "projects": project_poles
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[{SERVERS[server]}] 저장 완료: {filename}")
    return filename

def get_all_servers_anal2_poles():
    """
    모든 서버에서 프로젝트별 2차 분석 완료 전주 ID를 조회하여 저장
    """
    all_servers_data = {}
    summary = {}
    
    for server in SERVERS.keys():
        try:
            # 프로젝트 목록 조회
            PDB.poledb_init(server)
            
            # 데이터베이스 연결 확인
            if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
                print(f"[{SERVERS[server]}] 데이터베이스 연결 실패")
                continue
            
            project_list = PDB.groupname_info()
            
            if project_list is None or len(project_list) == 0:
                print(f"[{SERVERS[server]}] 프로젝트 목록이 없습니다.")
                continue
            
            # 2차 분석 완료된 프로젝트만 필터링
            # 각 프로젝트에 대해 2차 분석 완료된 전주가 있는지 확인
            print(f"[{SERVERS[server]}] 2차 분석 완료된 프로젝트 필터링 중...")
            anal2_completed_projects = []
            for project_name in project_list:
                anal2_poles = get_anal2_completed_poles(server, project_name)
                if len(anal2_poles) > 0:
                    anal2_completed_projects.append(project_name)
            
            print(f"[{SERVERS[server]}] 2차 분석 완료 프로젝트: {len(anal2_completed_projects)}개 (전체: {len(project_list)}개)")
            
            if len(anal2_completed_projects) == 0:
                print(f"[{SERVERS[server]}] 2차 분석 완료된 프로젝트가 없습니다.")
                continue
            
            # 각 프로젝트의 2차 분석 완료 전주 조회
            project_poles = get_project_anal2_poles(server, anal2_completed_projects)
            
            if project_poles:
                # 개별 서버 파일 저장
                save_anal2_poles_to_json(server, project_poles)
                
                # 전체 통합 데이터에 추가
                total_poles = sum(info["pole_count"] for info in project_poles.values())
                all_servers_data[server] = {
                    "server_name": SERVERS[server],
                    "projects": project_poles,
                    "total_projects": len(project_poles),
                    "total_anal2_poles": total_poles
                }
                
                summary[server] = {
                    "projects": len(project_poles),
                    "poles": total_poles
                }
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 오류 발생: {e}")
            continue
    
    # 전체 통합 파일 저장
    if all_servers_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "2. anal2_pole_list")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        all_total_projects = sum(s["total_projects"] for s in all_servers_data.values())
        all_total_poles = sum(s["total_anal2_poles"] for s in all_servers_data.values())
        
        all_filename = f"{output_dir}/anal2_poles_all_{timestamp}.json"
        with open(all_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "servers": all_servers_data,
                "summary": {
                    "by_server": summary,
                    "total_projects": all_total_projects,
                    "total_anal2_poles": all_total_poles
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n전체 통합 파일 저장 완료: {all_filename}")
        
        # 요약 출력
        print("\n=== 서버별 2차 분석 완료 전주 통계 ===")
        for server, info in summary.items():
            print(f"{SERVERS[server]}: {info['projects']}개 프로젝트, {info['poles']}개 전주")
        print(f"전체: {all_total_projects}개 프로젝트, {all_total_poles}개 전주")
        
        return all_filename
    
    return None

if __name__ == "__main__":
    print("=" * 50)
    print("2차 분석 완료 전주 ID 조회 및 저장 시작")
    print("=" * 50)
    
    # 모든 서버에서 2차 분석 완료 전주 조회 및 저장
    get_all_servers_anal2_poles()
    
    print("\n" + "=" * 50)
    print("2차 분석 완료 전주 ID 조회 및 저장 완료")
    print("=" * 50)

