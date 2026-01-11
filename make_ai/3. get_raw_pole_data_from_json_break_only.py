#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anal2_poles_all JSON 파일에서 전주 목록을 읽어서 파단 전주의 원본 데이터만 조회하여 저장하는 스크립트
- 파단(B) 전주만 저장
- 파단 전주의 파단 위치 정보와 함께 저장
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

def get_pole_anal2_result_break(server, project_name, poleid):
    """
    전주의 2차 분석 결과 중 파단(B)인 경우만 조회
    - analstep=2 (2차 분석)만 조회
    - breakstate='B' (파단)만 조회
    - 데이터 검증 강화
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
        poleid: 전주 ID
    
    Returns:
        dict: 파단 분석 결과 정보 (breakstate, breakheight, breakdegree) 또는 None
    """
    try:
        # 데이터베이스 연결 확인
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{poleid}]: 데이터베이스 연결이 없습니다.")
            return None
        
        # 데이터베이스에서 직접 2차 분석 결과 조회 (analstep=2, breakstate='B')
        # tb_anal_result 테이블에서 조회
        
        # SQL 쿼리: 2차 분석(analstep=2)이고 파단(B)인 결과만 조회
        # tb_anal_state와 조인하여 groupname 확인
        query = """
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
        
        # poledb_conn에 직접 접근하여 조회
        result = PDB.poledb_conn.do_select_pd(query, data)
        
        # 데이터 검증
        if result is None or result.empty:
            return None
        
        # 첫 번째 행 가져오기
        row = result.iloc[0]
        
        # 데이터 검증: breakstate가 'B'인지 재확인
        breakstate = str(row.get('breakstate', '')).strip().upper()
        if breakstate != 'B':
            print(f"    경고 [{poleid}]: breakstate가 'B'가 아닙니다. (실제 값: {breakstate})")
            return None
        
        # 전주 ID 검증 (대소문자 무시)
        result_poleid = str(row.get('poleid', '')).strip()
        if result_poleid.upper() != str(poleid).strip().upper():
            print(f"    경고 [{poleid}]: 전주 ID 불일치 (조회된 ID: {result_poleid})")
            return None
        
        # 프로젝트 이름 검증
        result_groupname = str(row.get('groupname', '')).strip()
        if result_groupname != project_name:
            print(f"    경고 [{poleid}]: 프로젝트 이름 불일치 (조회된 프로젝트: {result_groupname}, 요청 프로젝트: {project_name})")
            return None
        
        # 파단 정보 추출
        breakheight = row.get('breakheight')
        breakdegree = row.get('breakdegree')
        
        # breakheight와 breakdegree가 NaN이 아닌지 확인
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
        
    except Exception as e:
        print(f"    [{poleid}] 분석 결과 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def scan_existing_poles(output_base_dir):
    """
    출력 디렉토리에서 이미 존재하는 전주 데이터를 스캔하여 세트로 반환
    (성능 최적화: 시작 시 한 번만 스캔하고 메모리에 캐싱)
    
    Args:
        output_base_dir: 출력 기본 디렉토리
    
    Returns:
        set: (project_name, poleid) 튜플의 세트
    """
    existing_poles = set()
    
    if not os.path.exists(output_base_dir):
        return existing_poles
    
    print(f"  기존 데이터 스캔 중: {output_base_dir}")
    
    # 먼저 프로젝트 목록과 각 프로젝트의 전주 목록 수집 (진행도 표시를 위해)
    projects_poles = []
    try:
        for project_name in os.listdir(output_base_dir):
            project_dir = os.path.join(output_base_dir, project_name)
            if os.path.isdir(project_dir):
                pole_list = []
                for poleid in os.listdir(project_dir):
                    pole_dir = os.path.join(project_dir, poleid)
                    if os.path.isdir(pole_dir):
                        pole_list.append((project_name, poleid, pole_dir))
                if pole_list:
                    projects_poles.append((project_name, pole_list))
    except (OSError, PermissionError) as e:
        print(f"  경고: 기존 데이터 스캔 중 오류 발생: {e}")
        return existing_poles
    
    # 전체 전주 수 계산
    total_poles = sum(len(poles) for _, poles in projects_poles)
    
    if total_poles == 0:
        print(f"  기존 데이터 스캔 완료: 0개 전주 발견")
        return existing_poles
    
    # 진행도 표시와 함께 스캔
    with tqdm(total=total_poles, desc="  기존 데이터 스캔", unit="전주") as pbar:
        for project_name, pole_list in projects_poles:
            for proj_name, poleid, pole_dir in pole_list:
                # 폴더 내에 CSV 파일이 있는지 확인
                try:
                    csv_files = [f for f in os.listdir(pole_dir) if f.endswith('.csv')]
                    if len(csv_files) > 0:
                        existing_poles.add((proj_name, poleid))
                except (OSError, PermissionError):
                    # 읽기 권한이 없거나 접근할 수 없는 경우 스킵
                    pass
                pbar.update(1)
                pbar.set_postfix(발견=f"{len(existing_poles)}개")
    
    print(f"  기존 데이터 스캔 완료: {len(existing_poles)}개 전주 발견 (전체 {total_poles}개 중)")
    return existing_poles

def check_pole_data_exists(existing_poles_set, project_name, poleid):
    """
    파단 전주 데이터가 이미 저장되어 있는지 확인 (메모리 캐시 사용)
    
    Args:
        existing_poles_set: scan_existing_poles()로 미리 스캔한 세트
        project_name: 프로젝트 이름
        poleid: 전주 ID
    
    Returns:
        bool: 데이터가 이미 존재하면 True
    """
    return (project_name, poleid) in existing_poles_set

def save_break_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir, existing_poles_set):
    """
    파단 전주의 원본 데이터를 조회하여 저장
    
    Args:
        server: 서버 이름
        project_name: 프로젝트 이름
        poleid: 전주 ID
        anal_result: 분석 결과 정보
        output_base_dir: 출력 기본 디렉토리
        existing_poles_set: 미리 스캔한 기존 전주 세트 (성능 최적화)
    
    Returns:
        bool or None: 저장 성공(True), 저장 실패(False), 이미 존재(None)
    """
    try:
        # 이미 데이터가 존재하는지 확인 (메모리 캐시 사용)
        data_exists = check_pole_data_exists(existing_poles_set, project_name, poleid)
        
        # 이미 데이터가 존재하면 건너뛰기
        if data_exists:
            print(f"    [{poleid}] 건너뛰기: 이미 데이터가 존재합니다.")
            return None  # 이미 존재함을 의미
        
        # 파단 정보 추출
        breakheight = anal_result.get('breakheight')
        breakdegree = anal_result.get('breakdegree')
        break_info = f"_breakheight_{breakheight}_breakdegree_{breakdegree}" if breakheight is not None and breakdegree is not None else ""
        
        # 출력 디렉토리 생성
        pole_dir = os.path.join(output_base_dir, project_name, poleid)
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
        
        # 파단 정보를 별도 JSON 파일로 저장 (measno별 측정 정보 포함)
        break_info_file = os.path.join(pole_dir, f"{poleid}_break_info.json")
        info_exists = os.path.exists(break_info_file)
        
        print(f"    [{poleid}] break_info.json {'업데이트' if info_exists else '저장'} 중: {break_info_file}")
        print(f"    [{poleid}] 측정 정보 개수: OUT {len([k for k in measurements_info.keys() if k.startswith('OUT_')])}개, IN {len([k for k in measurements_info.keys() if k.startswith('IN_')])}개")
        with open(break_info_file, 'w', encoding='utf-8') as f:
            json.dump({
                'poleid': poleid,
                'project_name': project_name,
                'breakstate': 'B',
                'breakheight': breakheight,
                'breakdegree': breakdegree,
                'measurements': measurements_info
            }, f, ensure_ascii=False, indent=2)
        print(f"    [{poleid}] break_info.json {'업데이트' if info_exists else '저장'} 완료")
        
        # 저장 성공 - 캐시에 추가
        csv_count = len([f for f in os.listdir(pole_dir) if f.endswith('.csv')])
        existing_poles_set.add((project_name, poleid))  # 캐시 업데이트
        print(f"    [{poleid}] 추가됨: CSV 파일 {csv_count}개 저장 완료 (breakheight={breakheight}, breakdegree={breakdegree})")
        return True
        
    except Exception as e:
        print(f"    [{poleid}] 원본 데이터 저장 오류: {e}")
        return False

def process_break_poles_from_json(json_file_path):
    """
    JSON 파일에서 전주 목록을 읽어서 파단 전주의 원본 데이터만 조회하여 저장
    
    Args:
        json_file_path: JSON 파일 경로
    """
    # JSON 파일 읽기
    print(f"\nJSON 파일 읽기: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 디렉토리 생성 (파단 전주만 저장)
    output_base_dir = os.path.join(current_dir, "3. raw_pole_data", "break")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 기존 데이터 스캔 (성능 최적화: 시작 시 한 번만 스캔)
    print(f"\n{'='*60}")
    print("기존 데이터 스캔 중...")
    print(f"{'='*60}")
    existing_poles_set = scan_existing_poles(output_base_dir)
    
    # 통계 정보
    stats = {
        'total_poles': 0,
        'break_poles': 0,
        'skipped_poles': 0,  # 파단이 아닌 전주
        'skipped_existing': 0,  # 이미 존재하는 전주
        'error_poles': 0
    }
    
    # 각 서버별로 처리
    servers_data = data.get('servers', {})
    
    for server, server_info in servers_data.items():
        # JT 서버 데이터는 사용하지 않음
        if server == "jt":
            print(f"\n[JT 서버] 데이터 수집 대상에서 제외합니다.")
            continue
        print(f"\n{'='*60}")
        print(f"[{SERVERS[server]}] 처리 시작 (파단 전주만)")
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
                
                print(f"  전체 전주 개수: {total_poles_in_project}개")
                print(f"  파단 전주만 필터링 중...")
                
                # 각 전주 처리
                break_count = 0
                for pole_idx, poleid in enumerate(tqdm(pole_ids, desc=f"  {project_name} 처리 중"), 1):
                    stats['total_poles'] += 1
                    
                    try:
                        # 2차 분석 결과 조회 (파단만)
                        anal_result = get_pole_anal2_result_break(server, project_name, poleid)
                        
                        if anal_result is None:
                            # 파단이 아닌 경우 (저장하지 않음)
                            stats['skipped_poles'] += 1
                            continue
                        
                        # 파단 전주 원본 데이터 저장
                        result = save_break_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir, existing_poles_set)
                        
                        if result is None:
                            # 이미 존재하는 경우
                            stats['skipped_existing'] += 1
                            break_count += 1  # 파단 전주로 카운트
                        elif result:
                            # 저장 성공
                            stats['break_poles'] += 1
                            break_count += 1
                        else:
                            # 저장 실패
                            print(f"    [{poleid}] 오류: 데이터 저장 실패")
                            stats['error_poles'] += 1
                    
                    except Exception as e:
                        print(f"    [{poleid}] 오류: 처리 중 예외 발생 - {e}")
                        stats['error_poles'] += 1
                        continue
                
                print(f"  파단 전주: {break_count}개")
            
            print(f"\n[{SERVERS[server]}] 처리 완료")
        
        except Exception as e:
            print(f"[{SERVERS[server]}] 서버 처리 오류: {e}")
            continue
    
    # 최종 통계 출력
    print(f"\n{'='*60}")
    print("전체 처리 완료 (파단 전주만)")
    print(f"{'='*60}")
    print(f"전체 전주 수: {stats['total_poles']}개")
    print(f"파단 전주 (추가됨): {stats['break_poles']}개")
    print(f"파단 전주 (건너뛰기 - 이미 존재): {stats['skipped_existing']}개")
    print(f"제외된 전주 (파단 아님): {stats['skipped_poles']}개")
    print(f"오류 발생 전주: {stats['error_poles']}개")
    print(f"\n요약:")
    print(f"  - 새로 추가된 파단 전주: {stats['break_poles']}개")
    print(f"  - 이미 존재하여 건너뛴 파단 전주: {stats['skipped_existing']}개")
    print(f"  - 총 파단 전주: {stats['break_poles'] + stats['skipped_existing']}개")
    print(f"\n저장 위치: {output_base_dir}")
    print(f"  - 파단 전주만 저장: {output_base_dir}/")

if __name__ == "__main__":
    print("=" * 60)
    print("파단 전주 원본 데이터 수집 시작")
    print("=" * 60)
    
    try:
        # 최신 JSON 파일 찾기
        json_file = find_latest_json_file()
        
        # 파단 전주 데이터 처리
        process_break_poles_from_json(json_file)
        
        print("\n" + "=" * 60)
        print("파단 전주 원본 데이터 수집 완료")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

