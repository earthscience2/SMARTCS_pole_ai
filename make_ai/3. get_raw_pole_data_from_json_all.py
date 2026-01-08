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

import poledb as PDB

# 서버 정보
SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
    "jt": "제이티엔지니어링"
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
        
        # 데이터가 없을 때만 CSV 파일 저장
        if not data_exists:
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
        
        # 각 measno별 측정 정보 수집 (항상 수행)
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
        
        # 정보를 별도 JSON 파일로 저장 (항상 저장/업데이트)
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
        
        # 데이터가 이미 존재했으면 None 반환, 새로 저장했으면 True 반환
        if data_exists:
            return None  # 이미 존재함을 의미
        else:
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
        'error_poles': 0
    }
    
    # 각 서버별로 처리
    servers_data = data.get('servers', {})
    
    for server, server_info in servers_data.items():
        print(f"\n{'='*60}")
        print(f"[{SERVERS[server]}] 처리 시작")
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
                
                print(f"  전주 개수: {total_poles_in_project}개")
                
                # 각 전주 처리
                for pole_idx, poleid in enumerate(tqdm(pole_ids, desc=f"  {project_name} 처리 중"), 1):
                    stats['total_poles'] += 1
                    
                    try:
                        # 2차 분석 결과 조회
                        anal_result = get_pole_anal2_result(server, project_name, poleid)
                        
                        if anal_result is None:
                            # B나 N이 아닌 경우 (저장하지 않음)
                            stats['skipped_poles'] += 1
                            continue
                        
                        # 원본 데이터 저장
                        result = save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir)
                        
                        if result is None:
                            # 이미 존재하는 경우
                            stats['skipped_existing'] += 1
                        elif result:
                            # 저장 성공
                            if anal_result['breakstate'] == 'B':
                                stats['break_poles'] += 1
                            else:
                                stats['normal_poles'] += 1
                        else:
                            # 저장 실패
                            stats['error_poles'] += 1
                    
                    except Exception as e:
                        print(f"    [{poleid}] 처리 오류: {e}")
                        stats['error_poles'] += 1
                        continue
            
            print(f"\n[{SERVERS[server]}] 처리 완료")
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 서버 처리 오류: {e}")
            continue
    
    # 최종 통계 출력
    print(f"\n{'='*60}")
    print("전체 처리 완료")
    print(f"{'='*60}")
    print(f"전체 전주 수: {stats['total_poles']}개")
    print(f"파단 전주: {stats['break_poles']}개")
    print(f"정상 전주: {stats['normal_poles']}개")
    print(f"제외된 전주 (B/N 외): {stats['skipped_poles']}개")
    print(f"이미 존재하는 전주 (건너뜀): {stats['skipped_existing']}개")
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

