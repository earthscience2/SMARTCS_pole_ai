#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
각 프로젝트별로 2차 분석까지 완료된 전주 ID를 조회하여 정리하는 스크립트
"""

import sys
import os
import json
import glob
import pandas as pd
from datetime import datetime
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

def find_latest_project_list_json(server=None):
    """
    1. project_info_list 폴더에서 가장 최신의 프로젝트 목록 JSON 파일을 찾기
    
    Args:
        server: 서버 이름 (지정하면 개별 서버 파일도 찾음)
    
    Returns:
        str: 최신 JSON 파일 경로
    """
    json_dir = os.path.join(current_dir, "1. project_info_list")
    
    # 먼저 project_list_all 파일 찾기
    pattern_all = os.path.join(json_dir, "project_list_all_*.json")
    json_files_all = glob.glob(pattern_all)
    
    # 개별 서버 파일도 찾기 (server가 지정된 경우)
    json_files = json_files_all[:]
    if server:
        pattern_server = os.path.join(json_dir, f"project_list_{server}_*.json")
        json_files_server = glob.glob(pattern_server)
        json_files.extend(json_files_server)
    
    if not json_files:
        if server:
            raise FileNotFoundError(f"프로젝트 목록 JSON 파일을 찾을 수 없습니다: {pattern_all} 또는 project_list_{server}_*.json")
        else:
            raise FileNotFoundError(f"project_list_all JSON 파일을 찾을 수 없습니다: {pattern_all}")
    
    # 파일명의 타임스탬프를 기준으로 정렬 (가장 최신 파일)
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"최신 프로젝트 목록 JSON 파일: {latest_file}")
    return latest_file

def get_project_list_from_json(json_file_path, server):
    """
    JSON 파일에서 특정 서버의 프로젝트 목록을 추출
    - project_list_all_*.json 형식 (servers 키 아래): 지원
    - project_list_{server}_*.json 형식 (projects 키 직접): 지원
    
    Args:
        json_file_path: JSON 파일 경로
        server: 서버 이름
    
    Returns:
        list: 프로젝트 이름 목록 (2차 분석 완료된 프로젝트만, anal2_completed > 0)
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    projects = None
    
    # 형식 1: project_list_all_*.json (servers 키 아래)
    if 'servers' in data:
        servers_data = data.get('servers', {})
        server_data = servers_data.get(server)
        if server_data:
            projects = server_data.get('projects', [])
    
    # 형식 2: project_list_{server}_*.json (projects 키 직접)
    elif 'projects' in data:
        # 파일명에서 서버 이름 확인
        file_name = os.path.basename(json_file_path)
        if f"project_list_{server}_" in file_name:
            projects = data.get('projects', [])
    
    if projects is None:
        print(f"    경고: {SERVERS[server]}의 데이터를 찾을 수 없습니다. (파일: {os.path.basename(json_file_path)})")
        return []
    
    # 2차 분석 완료된 프로젝트만 필터링 (anal2_completed > 0)
    project_list = []
    for project in projects:
        project_name = project.get('project_name')
        statistics = project.get('statistics', {})
        anal2_completed = statistics.get('anal2_completed', 0)
        
        if project_name and anal2_completed > 0:
            project_list.append(project_name)
    
    return project_list

def get_anal2_completed_poles(server, project_name):
    """
    프로젝트에서 2차 분석까지 완료된 전주 ID 목록을 조회
    - anal2finyn이 NULL이 아닌 경우 (2차 분석 완료 표시)
    - tb_anal_result에 analstep=2인 결과가 있는지 확인
    - 각 전주의 파단/정상 여부(breakstate)도 함께 조회
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
    
    Returns:
        list: 2차 분석 완료된 전주 정보 목록 (각 항목은 {"poleid": str, "breakstate": str} 형식)
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{project_name}]: 데이터베이스 연결이 없습니다.")
            return []
        
        # 2차 분석 완료된 전주 조회 (breakstate 정보 포함)
        # 1. tb_anal_state에서 anal2finyn이 NULL이 아닌 전주
        # 2. tb_anal_result에 analstep=2인 결과가 있는 전주
        # 두 조건을 모두 만족하는 전주만 반환
        # breakstate는 각 전주별로 최신 결과(regdate DESC)를 기준으로 가져옴
        # 최적화: 서브쿼리 대신 JOIN과 랭킹을 사용하여 성능 개선
        query = """
            SELECT 
                tas.poleid,
                COALESCE(tar.breakstate, 'N') as breakstate
            FROM tb_anal_state tas
            INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
            INNER JOIN (
                SELECT 
                    poleid,
                    MAX(regdate) as max_regdate
                FROM tb_anal_result
                WHERE analstep = 2
                GROUP BY poleid
            ) tar_max ON tar.poleid = tar_max.poleid 
                AND tar.regdate = tar_max.max_regdate
                AND tar.analstep = 2
            WHERE tas.groupname = %s
            AND tas.anal2finyn IS NOT NULL
            GROUP BY tas.poleid, tar.breakstate
            ORDER BY tas.poleid
        """
        
        data = [project_name]
        
        # poledb_conn에 직접 접근
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        if result is None or result.empty:
            return []
        
        # 각 전주에 대해 poleid와 breakstate를 딕셔너리로 변환
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
        print(f"    [{project_name}] 2차 분석 완료 전주 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_project_anal2_poles(server, project_list):
    """
    서버의 모든 프로젝트에서 2차 분석 완료된 전주 정보를 조회 (breakstate 포함)
    
    Args:
        server: 서버 이름
        project_list: 프로젝트 목록
    
    Returns:
        dict: 프로젝트별 2차 분석 완료 전주 정보 목록 (breakstate 포함)
    """
    print(f"\n[{SERVERS[server]}] 연결 중...")
    PDB.poledb_init(server)
    
    project_poles = {}
    
    print(f"[{SERVERS[server]}] 전체 프로젝트 개수: {len(project_list)}개")
    print(f"[{SERVERS[server]}] 각 프로젝트의 2차 분석 완료 전주 조회 중...")
    
    # 통계 정보
    break_count_total = 0
    normal_count_total = 0
    
    # 진행도 표시와 함께 프로젝트 처리
    for project_name in tqdm(project_list, desc=f"  [{SERVERS[server]}] 프로젝트 처리", unit="프로젝트"):
        # 2차 분석 완료된 전주 정보 조회 (breakstate 포함)
        anal2_poles = get_anal2_completed_poles(server, project_name)
        
        # breakstate별 통계 계산 및 데이터 구조 변환
        break_count = sum(1 for pole in anal2_poles if pole.get('breakstate') == 'B')
        normal_count = sum(1 for pole in anal2_poles if pole.get('breakstate') == 'N')
        break_count_total += break_count
        normal_count_total += normal_count
        
        # 기존 호환성을 위해 pole_ids는 문자열 리스트로 유지
        pole_ids = [pole['poleid'] for pole in anal2_poles]
        
        # breakstate 정보를 별도 필드로 저장 (poleid -> breakstate 매핑)
        poles_info = {pole['poleid']: pole['breakstate'] for pole in anal2_poles}
        
        project_poles[project_name] = {
            "pole_count": len(anal2_poles),
            "break_count": break_count,
            "normal_count": normal_count,
            "pole_ids": pole_ids,  # 기존 호환성: 문자열 리스트
            "poles_info": poles_info  # breakstate 정보: {poleid: breakstate} 딕셔너리
        }
        
        # 진행도 바에 프로젝트별 통계 정보 표시
        tqdm.write(f"    [{project_name}] 완료 (전체: {len(anal2_poles)}개, 파단: {break_count}개, 정상: {normal_count}개)")
    
    print(f"[{SERVERS[server]}] 조회 완료 (전체: {break_count_total + normal_count_total}개, 파단: {break_count_total}개, 정상: {normal_count_total}개)")
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
    - 1. project_info_list의 JSON 파일에서 프로젝트 목록을 읽어옴
    """
    all_servers_data = {}
    summary = {}
    
    # 최신 프로젝트 목록 JSON 파일 찾기
    try:
        project_list_json_file = find_latest_project_list_json()
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("1. project_info_list 폴더에 project_list_all_*.json 파일이 필요합니다.")
        return None
    
    for server in SERVERS.keys():
        try:
            # JSON 파일에서 프로젝트 목록 읽기 시도
            print(f"\n[{SERVERS[server]}] 프로젝트 목록 읽는 중...")
            project_list = get_project_list_from_json(project_list_json_file, server)
            
            # 통합 파일에 서버 데이터가 없으면 개별 파일 찾기 시도
            if not project_list:
                try:
                    print(f"    통합 파일에 없음, 개별 서버 파일 찾는 중...")
                    individual_json_file = find_latest_project_list_json(server)
                    project_list = get_project_list_from_json(individual_json_file, server)
                except FileNotFoundError:
                    pass  # 개별 파일도 없으면 그냥 계속 진행
            
            if project_list is None or len(project_list) == 0:
                print(f"[{SERVERS[server]}] 2차 분석 완료된 프로젝트가 없습니다.")
                continue
            
            print(f"[{SERVERS[server]}] 2차 분석 완료 프로젝트: {len(project_list)}개")
            
            # 데이터베이스 연결 (전주 정보 조회를 위해 필요)
            PDB.poledb_init(server)
            
            # 데이터베이스 연결 확인
            if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
                print(f"[{SERVERS[server]}] 데이터베이스 연결 실패")
                continue
            
            # 각 프로젝트의 2차 분석 완료 전주 조회
            project_poles = get_project_anal2_poles(server, project_list)
            
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

