#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anal2_poles_all JSON 파일에서 전주 목록을 읽어서 원본 데이터를 조회하여 저장하는 스크립트
- 파단(B)과 정상(N) 전주만 저장
- 파단 전주의 경우 파단 위치 정보도 함께 저장
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

def find_latest_json_file():
    """
    anal2_pole_list 폴더에서 가장 최신의 anal2_poles_all JSON 파일을 찾기
    
    Returns:
        str: 최신 JSON 파일 경로
    """
    json_dir = os.path.join(current_dir, "2. anal2_pole_list")
    pattern = os.path.join(json_dir, "anal2_poles_all_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        raise FileNotFoundError(f"anal2_poles_all JSON 파일을 찾을 수 없습니다: {pattern}")
    
    # 파일명의 타임스탬프를 기준으로 정렬 (가장 최신 파일)
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"최신 JSON 파일: {latest_file}")
    return latest_file

def get_pole_anal2_result(server, project_name, poleid):
    """
    전주의 2차 분석 결과를 조회 (B: 파단, N: 정상)
    - analstep=2 (2차 분석)만 조회
    - breakstate='B' 또는 'N'만 조회
    - 데이터 검증 강화
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
        poleid: 전주 ID
    
    Returns:
        dict: 분석 결과 정보 (breakstate, breakheight, breakdegree) 또는 None
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{poleid}]: 데이터베이스 연결이 없습니다.")
            return None
        
        # 데이터베이스에서 직접 2차 분석 결과 조회 (analstep=2, breakstate='B' 또는 'N')
        # 먼저 파단(B) 조회
        query_break = """
            SELECT tar.poleid, tar.breakstate, tar.breakheight, tar.breakdegree, tas.groupname
            FROM tb_anal_result tar
            JOIN tb_anal_state tas ON tar.poleid = tas.poleid
            WHERE tar.poleid = %s 
            AND tar.analstep = 2 
            AND tar.breakstate = 'B'
            AND tas.groupname = %s
            ORDER BY tar.regdate DESC
            LIMIT 1
        """
        
        data = [poleid, project_name]
        result = PDB.poledb_conn.do_select_pd(query_break, data)
        
        if result is not None and not result.empty:
            row = result.iloc[0]
            breakstate = str(row.get('breakstate', '')).strip().upper()
            if breakstate == 'B':
                # 데이터 검증
                result_poleid = str(row.get('poleid', '')).strip()
                result_groupname = str(row.get('groupname', '')).strip()
                if result_poleid.upper() != str(poleid).strip().upper():
                    print(f"    경고 [{poleid}]: 전주 ID 불일치 (조회된 ID: {result_poleid})")
                    return None
                if result_groupname != project_name:
                    print(f"    경고 [{poleid}]: 프로젝트 이름 불일치 (조회된 프로젝트: {result_groupname}, 요청 프로젝트: {project_name})")
                    return None
                
                breakheight = row.get('breakheight')
                breakdegree = row.get('breakdegree')
                if pd.notna(breakheight):
                    breakheight = float(breakheight)
                else:
                    breakheight = None
                if pd.notna(breakdegree):
                    breakdegree = float(breakdegree)
                else:
                    breakdegree = None
                
                return {
                    'breakstate': 'B',
                    'breakheight': breakheight,
                    'breakdegree': breakdegree
                }
        
        # 정상(N) 조회
        query_normal = """
            SELECT tar.poleid, tar.breakstate, tar.breakheight, tar.breakdegree, tas.groupname
            FROM tb_anal_result tar
            JOIN tb_anal_state tas ON tar.poleid = tas.poleid
            WHERE tar.poleid = %s 
            AND tar.analstep = 2 
            AND tar.breakstate = 'N'
            AND tas.groupname = %s
            ORDER BY tar.regdate DESC
            LIMIT 1
        """
        
        result = PDB.poledb_conn.do_select_pd(query_normal, data)
        
        if result is not None and not result.empty:
            row = result.iloc[0]
            breakstate = str(row.get('breakstate', '')).strip().upper()
            if breakstate == 'N':
                # 데이터 검증
                result_poleid = str(row.get('poleid', '')).strip()
                result_groupname = str(row.get('groupname', '')).strip()
                if result_poleid.upper() != str(poleid).strip().upper():
                    print(f"    경고 [{poleid}]: 전주 ID 불일치 (조회된 ID: {result_poleid})")
                    return None
                if result_groupname != project_name:
                    print(f"    경고 [{poleid}]: 프로젝트 이름 불일치 (조회된 프로젝트: {result_groupname}, 요청 프로젝트: {project_name})")
                    return None
                
                return {
                    'breakstate': 'N',
                    'breakheight': None,
                    'breakdegree': None
                }
        
        return None  # B나 N이 아닌 경우 (저장하지 않음)
        
    except Exception as e:
        print(f"    [{poleid}] 분석 결과 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_pole_data_exists(output_base_dir, project_name, poleid, anal_result):
    """
    전주 데이터가 이미 저장되어 있는지 확인
    
    Args:
        output_base_dir: 출력 기본 디렉토리
        project_name: 프로젝트 이름
        poleid: 전주 ID
        anal_result: 분석 결과 정보
    
    Returns:
        bool: 데이터가 이미 존재하면 True
    """
    # 파단/정상에 따라 폴더 구분
    if anal_result['breakstate'] == 'B':
        category = 'break'
    else:
        category = 'normal'
    
    pole_dir = os.path.join(output_base_dir, category, project_name, poleid)
    
    # 폴더가 존재하고 파일이 있는지 확인
    if os.path.exists(pole_dir) and os.path.isdir(pole_dir):
        # 폴더 내에 CSV 파일이 있는지 확인
        csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
        if len(csv_files) > 0:
            return True
    
    return False

def count_saved_poles_in_project(output_base_dir, project_name, category):
    """
    프로젝트에 저장된 전주 수를 카운트
    
    Args:
        output_base_dir: 출력 기본 디렉토리
        project_name: 프로젝트 이름
        category: 'break' 또는 'normal'
    
    Returns:
        int: 저장된 전주 수
    """
    project_dir = os.path.join(output_base_dir, category, project_name)
    
    if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
        return 0
    
    # 프로젝트 디렉토리 내의 전주 디렉토리 수 카운트 (CSV 파일이 있는 것만)
    count = 0
    for poleid_dir in os.listdir(project_dir):
        pole_dir = os.path.join(project_dir, poleid_dir)
        if os.path.isdir(pole_dir):
            # CSV 파일이 있는지 확인
            csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
            if len(csv_files) > 0:
                count += 1
    
    return count

def count_all_saved_poles_in_project(output_base_dir, project_name):
    """
    프로젝트에 저장된 전체 전주 수를 카운트 (파단 + 정상)
    
    Args:
        output_base_dir: 출력 기본 디렉토리
        project_name: 프로젝트 이름
    
    Returns:
        int: 저장된 전체 전주 수 (break + normal)
    """
    break_count = count_saved_poles_in_project(output_base_dir, project_name, 'break')
    normal_count = count_saved_poles_in_project(output_base_dir, project_name, 'normal')
    return break_count + normal_count

def get_saved_pole_ids_in_project(output_base_dir, project_name):
    """
    프로젝트에 저장된 전주 ID 목록을 반환 (파단 + 정상)
    
    Args:
        output_base_dir: 출력 기본 디렉토리
        project_name: 프로젝트 이름
    
    Returns:
        set: 저장된 전주 ID 집합
    """
    saved_pole_ids = set()
    
    # break 폴더에서 전주 ID 수집
    break_dir = os.path.join(output_base_dir, 'break', project_name)
    if os.path.exists(break_dir) and os.path.isdir(break_dir):
        for poleid_dir in os.listdir(break_dir):
            pole_dir = os.path.join(break_dir, poleid_dir)
            if os.path.isdir(pole_dir):
                # CSV 파일이 있는지 확인
                csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
                if len(csv_files) > 0:
                    saved_pole_ids.add(poleid_dir)
    
    # normal 폴더에서 전주 ID 수집
    normal_dir = os.path.join(output_base_dir, 'normal', project_name)
    if os.path.exists(normal_dir) and os.path.isdir(normal_dir):
        for poleid_dir in os.listdir(normal_dir):
            pole_dir = os.path.join(normal_dir, poleid_dir)
            if os.path.isdir(pole_dir):
                # CSV 파일이 있는지 확인
                csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
                if len(csv_files) > 0:
                    saved_pole_ids.add(poleid_dir)
    
    return saved_pole_ids

def count_db_poles_in_project(server, project_name, breakstate):
    """
    DB에서 프로젝트의 전주 수를 조회 (최신 결과 기준)
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
        breakstate: 'B' (파단) 또는 'N' (정상)
    
    Returns:
        int: DB에 있는 전주 수
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            return 0
        
        # 2차 분석 완료된 전주 조회 (breakstate별, 최신 결과 기준)
        # get_anal2_completed_poles와 동일한 로직 사용
        query = """
            SELECT COUNT(DISTINCT tas.poleid) as count
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
            AND COALESCE(tar.breakstate, 'N') = %s
        """
        
        data = [project_name, breakstate]
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        if result is None or result.empty:
            return 0
        
        return int(result.iloc[0]['count'])
        
    except Exception as e:
        print(f"    [디버그] count_db_poles_in_project 오류: {e}")
        import traceback
        traceback.print_exc()
        return 0

def count_all_db_poles_in_project(server, project_name):
    """
    DB에서 프로젝트의 전체 전주 수를 조회 (파단 + 정상, 최신 결과 기준)
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
    
    Returns:
        int: DB에 있는 전체 전주 수
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            return 0
        
        # 2차 분석 완료된 전체 전주 조회 (최신 결과 기준)
        # get_anal2_completed_poles와 동일한 로직 사용
        query = """
            SELECT COUNT(DISTINCT tas.poleid) as count
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
        """
        
        data = [project_name]
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        if result is None or result.empty:
            return 0
        
        return int(result.iloc[0]['count'])
        
    except Exception as e:
        print(f"    [디버그] count_all_db_poles_in_project 오류: {e}")
        import traceback
        traceback.print_exc()
        return 0

def save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir):
    """
    전주의 원본 데이터를 조회하여 저장
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
        poleid: 전주 ID
        anal_result: 분석 결과 정보
        output_base_dir: 출력 기본 디렉토리
    
    Returns:
        bool or None: 저장 성공(True), 저장 실패(False), 이미 존재(None)
    """
    try:
        # 이미 데이터가 존재하는지 확인
        data_exists = check_pole_data_exists(output_base_dir, project_name, poleid, anal_result)
        
        # 데이터가 이미 존재하면 조기 반환 (DB 조회 및 JSON 저장도 건너뛰기)
        if data_exists:
            return None  # 이미 존재함을 의미
        
        # 파단/정상에 따라 폴더 구분
        if anal_result['breakstate'] == 'B':
            category = 'break'
            breakheight = anal_result.get('breakheight')
            breakdegree = anal_result.get('breakdegree')
            break_info = f"_breakheight_{breakheight}_breakdegree_{breakdegree}" if breakheight is not None and breakdegree is not None else ""
        else:
            category = 'normal'
            break_info = ""
        
        # 출력 디렉토리 생성
        pole_dir = os.path.join(output_base_dir, category, project_name, poleid)
        os.makedirs(pole_dir, exist_ok=True)
        
        # 측정 결과 정보 조회
        re_out = PDB.get_meas_result(poleid, 'OUT')
        re_in = PDB.get_meas_result(poleid, 'IN')
        
        num_sig_out = re_out.shape[0] if re_out is not None and not re_out.empty else 0
        num_sig_in = re_in.shape[0] if re_in is not None and not re_in.empty else 0
        
        # CSV 파일 저장
        # IN 데이터 저장
        for kk in range(num_sig_in):
            stype = 'IN'
            num = int(re_in['measno'][kk])
            time = str(re_in['sttime'][kk])
            time = (time.split(" "))[0] if " " in time else time
            
            in_x = PDB.get_meas_data(poleid, num, stype, 'x')
            if in_x is not None and not in_x.empty:
                filename = f"{poleid}_{kk+1}_{time}_IN_x{break_info}.csv"
                in_x.to_csv(os.path.join(pole_dir, filename), index=False)
        
        # OUT 데이터 저장
        for kk in range(num_sig_out):
            stype = 'OUT'
            num = int(re_out['measno'][kk])
            time = str(re_out['sttime'][kk])
            time = (time.split(" "))[0] if " " in time else time
            
            out_x = PDB.get_meas_data(poleid, num, stype, 'x')
            out_y = PDB.get_meas_data(poleid, num, stype, 'y')
            out_z = PDB.get_meas_data(poleid, num, stype, 'z')
            
            if out_x is not None and not out_x.empty:
                filename = f"{poleid}_{kk+1}_{time}_OUT_x{break_info}.csv"
                out_x.to_csv(os.path.join(pole_dir, filename), index=False)
            
            if out_y is not None and not out_y.empty:
                filename = f"{poleid}_{kk+1}_{time}_OUT_y{break_info}.csv"
                out_y.to_csv(os.path.join(pole_dir, filename), index=False)
            
            if out_z is not None and not out_z.empty:
                filename = f"{poleid}_{kk+1}_{time}_OUT_z{break_info}.csv"
                out_z.to_csv(os.path.join(pole_dir, filename), index=False)
        
        # 각 measno별 측정 정보 수집
        measurements_info = {}
        
        # 안전하게 컬럼 값 가져오기 (대소문자 구분)
        def safe_get_value(row, col_name, alt_names=None):
            """DataFrame row에서 컬럼 값을 안전하게 가져오기"""
            if col_name in row.index:
                val = row[col_name]
                if pd.notna(val):
                    return val
            if alt_names:
                for alt in alt_names:
                    if alt in row.index:
                        val = row[alt]
                        if pd.notna(val):
                            return val
            return None
        
        # OUT 측정 정보 수집
        if re_out is not None and not re_out.empty:
            for kk in range(num_sig_out):
                measno = int(re_out['measno'][kk])
                row = re_out.iloc[kk]
                
                meas_info = {
                    'measno': measno,
                    'devicetype': 'OUT',
                    'stdegree': float(safe_get_value(row, 'stdegree', ['stDegree'])) if safe_get_value(row, 'stdegree', ['stDegree']) is not None else None,
                    'eddegree': float(safe_get_value(row, 'eddegree', ['edDegree'])) if safe_get_value(row, 'eddegree', ['edDegree']) is not None else None,
                    'stheight': float(safe_get_value(row, 'stheight', ['stHeight'])) if safe_get_value(row, 'stheight', ['stHeight']) is not None else None,
                    'edheight': float(safe_get_value(row, 'edheight', ['edHeight'])) if safe_get_value(row, 'edheight', ['edHeight']) is not None else None,
                    'sttime': str(safe_get_value(row, 'sttime', ['stTime'])) if safe_get_value(row, 'sttime', ['stTime']) is not None else None,
                    'endtime': str(safe_get_value(row, 'endtime', ['endTime', 'edtime'])) if safe_get_value(row, 'endtime', ['endTime', 'edtime']) is not None else None
                }
                measurements_info[f'OUT_{measno}'] = meas_info
        
        # IN 측정 정보 수집
        if re_in is not None and not re_in.empty:
            for kk in range(num_sig_in):
                measno = int(re_in['measno'][kk])
                row = re_in.iloc[kk]
                
                meas_info = {
                    'measno': measno,
                    'devicetype': 'IN',
                    'stdegree': float(safe_get_value(row, 'stdegree', ['stDegree'])) if safe_get_value(row, 'stdegree', ['stDegree']) is not None else None,
                    'eddegree': float(safe_get_value(row, 'eddegree', ['edDegree'])) if safe_get_value(row, 'eddegree', ['edDegree']) is not None else None,
                    'stheight': float(safe_get_value(row, 'stheight', ['stHeight'])) if safe_get_value(row, 'stheight', ['stHeight']) is not None else None,
                    'edheight': float(safe_get_value(row, 'edheight', ['edHeight'])) if safe_get_value(row, 'edheight', ['edHeight']) is not None else None,
                    'sttime': str(safe_get_value(row, 'sttime', ['stTime'])) if safe_get_value(row, 'sttime', ['stTime']) is not None else None,
                    'endtime': str(safe_get_value(row, 'endtime', ['endTime', 'edtime'])) if safe_get_value(row, 'endtime', ['endTime', 'edtime']) is not None else None
                }
                measurements_info[f'IN_{measno}'] = meas_info
        
        # 정보를 별도 JSON 파일로 저장
        if anal_result['breakstate'] == 'B':
            info_file = os.path.join(pole_dir, f"{poleid}_break_info.json")
            print(f"    [{poleid}] break_info.json 저장 중: {info_file}")
            print(f"    [{poleid}] 측정 정보 개수: OUT {len([k for k in measurements_info.keys() if k.startswith('OUT_')])}개, IN {len([k for k in measurements_info.keys() if k.startswith('IN_')])}개")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'poleid': poleid,
                    'project_name': project_name,
                    'breakstate': 'B',
                    'breakheight': anal_result.get('breakheight'),
                    'breakdegree': anal_result.get('breakdegree'),
                    'measurements': measurements_info
                }, f, ensure_ascii=False, indent=2)
            print(f"    [{poleid}] break_info.json 저장 완료")
        else:
            info_file = os.path.join(pole_dir, f"{poleid}_normal_info.json")
            print(f"    [{poleid}] normal_info.json 저장 중: {info_file}")
            print(f"    [{poleid}] 측정 정보 개수: OUT {len([k for k in measurements_info.keys() if k.startswith('OUT_')])}개, IN {len([k for k in measurements_info.keys() if k.startswith('IN_')])}개")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'poleid': poleid,
                    'project_name': project_name,
                    'breakstate': 'N',
                    'breakheight': None,
                    'breakdegree': None,
                    'measurements': measurements_info
                }, f, ensure_ascii=False, indent=2)
            print(f"    [{poleid}] normal_info.json 저장 완료")
        
        # 새로 저장했으므로 True 반환
        return True
        
    except Exception as e:
        print(f"    [{poleid}] 원본 데이터 저장 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_all_poles_from_json(json_file_path):
    """
    JSON 파일에서 전주 목록을 읽어서 원본 데이터를 조회하여 저장
    
    Args:
        json_file_path: JSON 파일 경로
    """
    # JSON 파일 읽기
    print(f"\nJSON 파일 읽기: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 디렉토리 생성
    output_base_dir = os.path.join(current_dir, "3. raw_pole_data")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 통계 정보
    stats = {
        'total_poles': 0,
        'break_poles': 0,
        'normal_poles': 0,
        'skipped_poles': 0,
        'skipped_existing': 0,  # 이미 존재하는 전주
        'skipped_limit': 0,     # 정상 데이터 개수 제한으로 인해 건너뛴 전주
        'error_poles': 0
    }
    
    # 각 서버별 데이터
    servers_data = data.get('servers', {})
    
    # ------------------------------------------------------------------
    # 1단계: 파단(B) 전주 데이터 먼저 수집
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1단계: 파단(B) 전주 데이터 수집 시작")
    print("=" * 60)
    
    for server, server_info in servers_data.items():
        # JT 서버 데이터는 사용하지 않음
        if server == "jt":
            print(f"\n[JT 서버] 데이터 수집 대상에서 제외합니다.")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{SERVERS[server]}] (1단계) 파단 전주 처리 시작")
        print(f"{'='*60}")
        
        try:
            # 서버 연결
            PDB.poledb_init(server)
            
            projects_data = server_info.get('projects', {})
            total_projects = len(projects_data)
            
            for project_idx, (project_name, project_info) in enumerate(projects_data.items(), 1):
                print(f"\n[{project_idx}/{total_projects}] 프로젝트: {project_name}")
                
                pole_ids = project_info.get('pole_ids', [])
                total_poles_in_project = len(pole_ids)
                
                # 저장된 파단 전주 수 확인
                saved_break_count = count_saved_poles_in_project(output_base_dir, project_name, 'break')
                
                # DB에서 파단 전주 수 조회
                db_break_count = count_db_poles_in_project(server, project_name, 'B')
                
                print(f"  전주 개수: {total_poles_in_project}개")
                print(f"  저장됨 (파단: {saved_break_count}개), DB (파단: {db_break_count}개)")
                
                # 저장된 파단 전주 수와 DB 파단 전주 수가 같으면 건너뛰기 (0개 == 0개도 포함)
                if saved_break_count == db_break_count:
                    print(f"  → 저장된 파단 전주 수({saved_break_count}개)와 DB 파단 전주 수({db_break_count}개)가 같아 건너뜁니다.")
                    continue
                
                # 저장된 파단 전주 ID 목록 가져오기
                break_dir = os.path.join(output_base_dir, 'break', project_name)
                saved_break_pole_ids = set()
                if os.path.exists(break_dir) and os.path.isdir(break_dir):
                    for poleid_dir in os.listdir(break_dir):
                        pole_dir = os.path.join(break_dir, poleid_dir)
                        if os.path.isdir(pole_dir):
                            csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
                            if len(csv_files) > 0:
                                saved_break_pole_ids.add(poleid_dir)
                
                # 저장되지 않은 전주만 필터링
                unsaved_pole_ids = [pid for pid in pole_ids if pid not in saved_break_pole_ids]
                
                if len(unsaved_pole_ids) == 0:
                    print(f"  → 모든 파단 전주가 이미 저장되어 있습니다.")
                    continue
                
                print(f"  → 저장되지 않은 전주: {len(unsaved_pole_ids)}개 (전체: {len(pole_ids)}개)")
                
                # 각 전주 처리 (파단만 저장, 저장되지 않은 것만)
                for pole_idx, poleid in enumerate(tqdm(unsaved_pole_ids, desc=f"  {project_name} (파단 수집)"), 1):
                    stats['total_poles'] += 1
                    
                    try:
                        # 2차 분석 결과 조회
                        anal_result = get_pole_anal2_result(server, project_name, poleid)
                        
                        if anal_result is None:
                            # B나 N이 아닌 경우 (저장하지 않음)
                            stats['skipped_poles'] += 1
                            continue
                        
                        # 파단 전주만 1단계에서 저장
                        if anal_result['breakstate'] != 'B':
                            continue
                        
                        # 원본 데이터 저장
                        result = save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir)
                        
                        if result is None:
                            # 이미 존재하는 경우
                            stats['skipped_existing'] += 1
                        elif result:
                            # 저장 성공
                            stats['break_poles'] += 1
                        else:
                            # 저장 실패
                            stats['error_poles'] += 1
                    
                    except Exception as e:
                        print(f"    [{poleid}] 처리 오류: {e}")
                        stats['error_poles'] += 1
                        continue
            
            print(f"\n[{SERVERS[server]}] (1단계) 파단 전주 처리 완료")
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 서버 처리 오류: {e}")
            continue
    
    # 전체 저장된 파단 전주 수 확인 (모든 프로젝트 합계)
    total_saved_break_count = 0
    total_saved_normal_count = 0
    
    for server, server_info in servers_data.items():
        if server == "jt":
            continue
        projects_data = server_info.get('projects', {})
        for project_name in projects_data.keys():
            total_saved_break_count += count_saved_poles_in_project(output_base_dir, project_name, 'break')
            total_saved_normal_count += count_saved_poles_in_project(output_base_dir, project_name, 'normal')
    
    # 이번 실행에서 새로 저장된 파단 전주 수
    new_break_poles = stats['break_poles']
    # 전체 파단 전주 수 (기존 + 신규)
    total_break_poles = total_saved_break_count
    
    # 파단 전주 개수 기준으로 정상 전주 최대 개수 결정 (10배)
    max_normal_poles = total_break_poles * 10
    
    print("\n" + "=" * 60)
    print("2단계: 정상(N) 전주 데이터 수집 시작")
    print("=" * 60)
    print(f"전체 저장된 파단 전주 개수: {total_break_poles}개")
    print(f"전체 저장된 정상 전주 개수: {total_saved_normal_count}개")
    print(f"정상 전주 최대 수집 개수 (파단의 10배): {max_normal_poles}개")
    
    # 전체 저장된 정상 전주 수가 파단 전주 수의 10배 이상이면 수집 중단
    if total_break_poles > 0 and total_saved_normal_count >= max_normal_poles:
        print(f"  → 전체 저장된 정상 전주 수({total_saved_normal_count}개)가 파단 전주 수({total_break_poles}개)의 10배 이상이므로 정상 전주 수집을 중단합니다.")
        return
    
    # 현재까지 수집된 정상 전주 개수 (이번 실행에서 새로 저장된 것 기준)
    current_normal_poles = total_saved_normal_count
    
    # ------------------------------------------------------------------
    # 2단계: 정상(N) 전주 데이터 수집 (최대 파단의 10배까지)
    # ------------------------------------------------------------------
    for server, server_info in servers_data.items():
        # JT 서버 데이터는 사용하지 않음
        if server == "jt":
            continue
        
        # 더 이상 수집할 수 있는 정상 전주가 없으면 종료
        if current_normal_poles >= max_normal_poles and max_normal_poles > 0:
            break
        
        print(f"\n{'='*60}")
        print(f"[{SERVERS[server]}] (2단계) 정상 전주 처리 시작")
        print(f"{'='*60}")
        
        try:
            # 서버 연결
            PDB.poledb_init(server)
            
            projects_data = server_info.get('projects', {})
            total_projects = len(projects_data)
            
            for project_idx, (project_name, project_info) in enumerate(projects_data.items(), 1):
                print(f"\n[{project_idx}/{total_projects}] 프로젝트: {project_name}")
                
                pole_ids = project_info.get('pole_ids', [])
                total_poles_in_project = len(pole_ids)
                
                # 저장된 정상 전주 수 확인
                saved_normal_count = count_saved_poles_in_project(output_base_dir, project_name, 'normal')
                
                # DB에서 정상 전주 수 조회
                db_normal_count = count_db_poles_in_project(server, project_name, 'N')
                
                print(f"  전주 개수: {total_poles_in_project}개")
                print(f"  저장됨 (정상: {saved_normal_count}개), DB (정상: {db_normal_count}개)")
                
                # 저장된 정상 전주 수와 DB 정상 전주 수가 같으면 건너뛰기 (0개 == 0개도 포함)
                if saved_normal_count == db_normal_count:
                    print(f"  → 저장된 정상 전주 수({saved_normal_count}개)와 DB 정상 전주 수({db_normal_count}개)가 같아 건너뜁니다.")
                    continue
                
                # 전체 저장된 정상 전주 수가 전체 파단 전주 수의 10배 이상이면 건너뛰기
                if total_break_poles > 0 and current_normal_poles >= max_normal_poles:
                    print(f"  → 전체 정상 전주 수({current_normal_poles}개)가 전체 파단 전주 수({total_break_poles}개)의 10배 이상이므로 건너뜁니다.")
                    continue
                
                # 저장된 정상 전주 ID 목록 가져오기
                normal_dir = os.path.join(output_base_dir, 'normal', project_name)
                saved_normal_pole_ids = set()
                if os.path.exists(normal_dir) and os.path.isdir(normal_dir):
                    for poleid_dir in os.listdir(normal_dir):
                        pole_dir = os.path.join(normal_dir, poleid_dir)
                        if os.path.isdir(pole_dir):
                            csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
                            if len(csv_files) > 0:
                                saved_normal_pole_ids.add(poleid_dir)
                
                # 저장되지 않은 전주만 필터링
                unsaved_pole_ids = [pid for pid in pole_ids if pid not in saved_normal_pole_ids]
                
                if len(unsaved_pole_ids) == 0:
                    print(f"  → 모든 정상 전주가 이미 저장되어 있습니다.")
                    continue
                
                print(f"  → 저장되지 않은 전주: {len(unsaved_pole_ids)}개 (전체: {len(pole_ids)}개)")
                
                # 각 전주 처리 (정상만, 제한 개수까지, 저장되지 않은 것만)
                for pole_idx, poleid in enumerate(tqdm(unsaved_pole_ids, desc=f"  {project_name} (정상 수집)"), 1):
                    # 전체 저장된 정상 전주 수가 전체 파단 전주 수의 10배 이상이면 루프 종료
                    if total_break_poles > 0 and current_normal_poles >= max_normal_poles:
                        break
                    
                    try:
                        # 2차 분석 결과 조회
                        anal_result = get_pole_anal2_result(server, project_name, poleid)
                        
                        if anal_result is None:
                            # B나 N이 아닌 경우 (저장하지 않음)
                            stats['skipped_poles'] += 1
                            continue
                        
                        # 정상 전주만 2단계에서 저장
                        if anal_result['breakstate'] != 'N':
                            continue
                        
                        # 원본 데이터 저장
                        result = save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir)
                        
                        if result is None:
                            # 이미 존재하는 경우
                            stats['skipped_existing'] += 1
                        elif result:
                            # 저장 성공
                            stats['normal_poles'] += 1
                            current_normal_poles += 1
                            
                            # 전체 저장된 정상 전주 수가 전체 파단 전주 수의 10배 이상이면 루프 종료
                            if total_break_poles > 0 and current_normal_poles >= max_normal_poles:
                                break
                        else:
                            # 저장 실패
                            stats['error_poles'] += 1
                    
                    except Exception as e:
                        print(f"    [{poleid}] 처리 오류: {e}")
                        stats['error_poles'] += 1
                        continue
            
            print(f"\n[{SERVERS[server]}] (2단계) 정상 전주 처리 완료")
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 서버 처리 오류: {e}")
            continue
    
    # 최종 통계 출력
    print(f"\n{'='*60}")
    print("전체 처리 완료")
    print(f"{'='*60}")
    print(f"전체 전주 수: {stats['total_poles']}개")
    print(f"파단 전주: {stats['break_poles']}개")
    print(f"정상 전주 (새로 저장됨): {stats['normal_poles']}개")
    print(f"정상 전주 최대 허용 개수(파단의 10배): {max_normal_poles}개")
    print(f"제외된 전주 (B/N 외): {stats['skipped_poles']}개")
    print(f"이미 존재하는 전주 (건너뜀): {stats['skipped_existing']}개")
    print(f"정상 전주 개수 제한으로 건너뜀: {stats['skipped_limit']}개")
    print(f"오류 발생 전주: {stats['error_poles']}개")
    print(f"\n저장 위치: {output_base_dir}")
    print(f"  - 파단 전주: {output_base_dir}/break/")
    print(f"  - 정상 전주: {output_base_dir}/normal/")

if __name__ == "__main__":
    print("=" * 60)
    print("원본 전주 데이터 수집 시작")
    print("=" * 60)
    
    try:
        # 최신 JSON 파일 찾기
        json_file = find_latest_json_file()
        
        # 전주 데이터 처리
        process_all_poles_from_json(json_file)
        
        print("\n" + "=" * 60)
        print("원본 전주 데이터 수집 완료")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

