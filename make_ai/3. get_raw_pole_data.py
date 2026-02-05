#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anal2_poles_all JSON에서 전주 목록을 읽어 원본 측정 데이터를 조회·저장"""

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

INPUT_DIR = "2. anal_pole_list"
OUTPUT_DIR = "3. raw_pole_data"
NORMAL_POLE_RATIO = 10  # 정상 전주 최대 개수 = 파단 × 이 비율


def find_latest_json_file():
    """2. anal_pole_list의 anal2_poles_all_*.json 통합 파일 중 가장 최신 파일 경로 반환"""
    json_dir = os.path.join(current_dir, INPUT_DIR)
    pattern = os.path.join(json_dir, "anal2_poles_all_*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        raise FileNotFoundError(f"anal2_poles_all JSON 파일을 찾을 수 없습니다: {pattern}")
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"최신 JSON 파일: {latest_file}")
    return latest_file

def _safe_float(val):
    """pd.notna 체크 후 float 변환"""
    return float(val) if val is not None and pd.notna(val) else None


def get_pole_anal2_result(server, project_name, poleid):
    """전주 분석 결과 조회 (B/N만, 2차 분석 우선)"""
    try:
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            print(f"    경고 [{poleid}]: 데이터베이스 연결이 없습니다.")
            return None

        data = [poleid, project_name]
        query_break = """
            SELECT 
                tar.poleid, 
                tar.breakstate, 
                tar.breakheight, 
                tar.breakdegree, 
                tar.analstep,
                tas.groupname
            FROM tb_anal_result tar
            JOIN tb_anal_state tas ON tar.poleid = tas.poleid
            WHERE tar.poleid = %s 
            AND tar.analstep IN (1, 2)
            AND tar.breakstate = 'B'
            AND tas.groupname = %s
            ORDER BY tar.analstep DESC, tar.regdate DESC
            LIMIT 1
        """
        result = PDB.poledb_conn.do_select_pd(query_break, data)
        if result is not None and not result.empty:
            row = result.iloc[0]
            if str(row.get('breakstate', '')).strip().upper() == 'B':
                if str(row.get('poleid', '')).strip().upper() != str(poleid).strip().upper():
                    return None
                if str(row.get('groupname', '')).strip() != project_name:
                    return None
                return {
                    'breakstate': 'B',
                    'breakheight': _safe_float(row.get('breakheight')),
                    'breakdegree': _safe_float(row.get('breakdegree')),
                }

        # 정상(N) 조회
        query_normal = """
            SELECT 
                tar.poleid, 
                tar.breakstate, 
                tar.breakheight, 
                tar.breakdegree, 
                tar.analstep,
                tas.groupname
            FROM tb_anal_result tar
            JOIN tb_anal_state tas ON tar.poleid = tas.poleid
            WHERE tar.poleid = %s 
            AND tar.analstep IN (1, 2)
            AND tar.breakstate = 'N'
            AND tas.groupname = %s
            ORDER BY tar.analstep DESC, tar.regdate DESC
            LIMIT 1
        """
        result = PDB.poledb_conn.do_select_pd(query_normal, data)
        if result is not None and not result.empty:
            row = result.iloc[0]
            if str(row.get('breakstate', '')).strip().upper() == 'N':
                if str(row.get('poleid', '')).strip().upper() != str(poleid).strip().upper():
                    return None
                if str(row.get('groupname', '')).strip() != project_name:
                    return None
                return {'breakstate': 'N', 'breakheight': None, 'breakdegree': None}

        return None
    except Exception as e:
        print(f"    [{poleid}] 분석 결과 조회 오류: {e}")
        traceback.print_exc()
        return None

def _pole_dir_has_csv(pole_dir):
    """전주 디렉토리에 CSV 파일이 있는지 확인"""
    if not os.path.isdir(pole_dir):
        return False
    return any(f.endswith('.csv') for f in os.listdir(pole_dir))


def _get_saved_pole_ids(output_base_dir, project_name, category):
    """카테고리별 저장된 전주 ID 집합 반환"""
    project_dir = os.path.join(output_base_dir, category, project_name)
    if not os.path.isdir(project_dir):
        return set()
    return {
        d for d in os.listdir(project_dir)
        if _pole_dir_has_csv(os.path.join(project_dir, d))
    }


def check_pole_data_exists(output_base_dir, project_name, poleid, anal_result):
    """전주 데이터 저장 여부 확인"""
    category = 'break' if anal_result['breakstate'] == 'B' else 'normal'
    pole_dir = os.path.join(output_base_dir, category, project_name, poleid)
    return _pole_dir_has_csv(pole_dir)


def count_saved_poles_in_project(output_base_dir, project_name, category):
    """프로젝트에 저장된 전주 수 (카테고리별)"""
    return len(_get_saved_pole_ids(output_base_dir, project_name, category))


def get_saved_pole_ids_in_project(output_base_dir, project_name):
    """프로젝트에 저장된 전체 전주 ID 집합 (break + normal)"""
    return _get_saved_pole_ids(output_base_dir, project_name, 'break') | \
           _get_saved_pole_ids(output_base_dir, project_name, 'normal')

_QUERY_DB_POLE_COUNT = """
    SELECT COUNT(DISTINCT tas.poleid) as count
    FROM tb_anal_state tas
    INNER JOIN tb_anal_result tar ON tas.poleid = tar.poleid
    INNER JOIN (
        SELECT tar1.poleid, MAX(tar1.analstep) as max_analstep
        FROM tb_anal_result tar1
        WHERE tar1.analstep IN (1, 2)
        GROUP BY tar1.poleid
    ) tar_max ON tar.poleid = tar_max.poleid
    INNER JOIN (
        SELECT tar2.poleid, tar2.analstep, MAX(tar2.regdate) as max_regdate
        FROM tb_anal_result tar2
        WHERE tar2.analstep IN (1, 2)
        GROUP BY tar2.poleid, tar2.analstep
    ) tar_latest ON tar.poleid = tar_latest.poleid
        AND tar.analstep = tar_latest.analstep
        AND tar.regdate = tar_latest.max_regdate
        AND tar.analstep = tar_max.max_analstep
    WHERE tas.groupname = %s
"""


def _count_db_poles(project_name, breakstate=None):
    """DB 전주 수 조회 (breakstate 지정 시 필터)"""
    try:
        if not hasattr(PDB, 'poledb_conn') or PDB.poledb_conn is None:
            return 0
        query = _QUERY_DB_POLE_COUNT + (" AND COALESCE(tar.breakstate, 'N') = %s" if breakstate else "")
        data = [project_name, breakstate] if breakstate else [project_name]
        result = PDB.poledb_conn.do_select_pd(query, data)
        return int(result.iloc[0]['count']) if result is not None and not result.empty else 0
    except Exception as e:
        print(f"    [디버그] DB 전주 수 조회 오류: {e}")
        traceback.print_exc()
        return 0


def count_db_poles_in_project(server, project_name, breakstate):
    """DB에서 프로젝트의 전주 수 조회 (B 또는 N, 2차 분석 우선)"""
    return _count_db_poles(project_name, breakstate)

def _safe_get_value(row, col_name, alt_names=None):
    """DataFrame row에서 컬럼 값을 안전하게 가져오기"""
    for name in ([col_name] + (alt_names or [])):
        if name in row.index and pd.notna(row.get(name)):
            return row[name]
    return None


def _build_meas_info(row):
    """측정 결과 row에서 meas_info 딕셔너리 생성"""
    def _f(col, alts):
        v = _safe_get_value(row, col, alts)
        return float(v) if v is not None else None
    def _s(col, alts):
        v = _safe_get_value(row, col, alts)
        return str(v) if v is not None else None
    return {
        'stdegree': _f('stdegree', ['stDegree']),
        'eddegree': _f('eddegree', ['edDegree']),
        'stheight': _f('stheight', ['stHeight']),
        'edheight': _f('edheight', ['edHeight']),
        'sttime': _s('sttime', ['stTime']),
        'endtime': _s('endtime', ['endTime', 'edtime']),
    }


def save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir):
    """전주 원본 데이터 조회 및 저장. 반환: True(성공), False(실패), None(이미 존재)"""
    try:
        if check_pole_data_exists(output_base_dir, project_name, poleid, anal_result):
            return None

        if anal_result['breakstate'] == 'B':
            category = 'break'
            breakheight = anal_result.get('breakheight')
            breakdegree = anal_result.get('breakdegree')
            break_info = f"_breakheight_{breakheight}_breakdegree_{breakdegree}" if breakheight is not None and breakdegree is not None else ""
        else:
            category = 'normal'
            break_info = ""
        
        pole_dir = os.path.join(output_base_dir, category, project_name, poleid)
        os.makedirs(pole_dir, exist_ok=True)

        # 측정 결과 조회
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
        
        measurements_info = {}
        if re_out is not None and not re_out.empty:
            for kk in range(num_sig_out):
                measno = int(re_out['measno'][kk])
                meas_info = {'measno': measno, 'devicetype': 'OUT', **_build_meas_info(re_out.iloc[kk])}
                measurements_info[f'OUT_{measno}'] = meas_info
        if re_in is not None and not re_in.empty:
            for kk in range(num_sig_in):
                measno = int(re_in['measno'][kk])
                meas_info = {'measno': measno, 'devicetype': 'IN', **_build_meas_info(re_in.iloc[kk])}
                measurements_info[f'IN_{measno}'] = meas_info

        # JSON 메타 정보 저장
        info_filename = f"{poleid}_break_info.json" if anal_result['breakstate'] == 'B' else f"{poleid}_normal_info.json"
        info_data = {
            'poleid': poleid,
            'project_name': project_name,
            'breakstate': anal_result['breakstate'],
            'breakheight': anal_result.get('breakheight') if anal_result['breakstate'] == 'B' else None,
            'breakdegree': anal_result.get('breakdegree') if anal_result['breakstate'] == 'B' else None,
            'measurements': measurements_info,
        }
        with open(os.path.join(pole_dir, info_filename), 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"    [{poleid}] 원본 데이터 저장 오류: {e}")
        traceback.print_exc()
        return False

def _save_run_summary_json(output_base_dir, json_file_path, stats, max_normal_poles, total_saved_break, total_saved_normal_after_run):
    """실행 요약을 통합 JSON 1개로 저장 (1·2번 스크립트 저장 방식에 맞춤)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_filename = os.path.join(output_base_dir, f"raw_pole_data_summary_{timestamp}.json")
    summary_data = {
        "timestamp": timestamp,
        "source_json": os.path.basename(json_file_path),
        "source_path": json_file_path,
        "output_dir": output_base_dir,
        "stats": stats,
        "max_normal_poles": max_normal_poles,
        "total_saved_break": total_saved_break,
        "total_saved_normal_after_run": total_saved_normal_after_run,
    }
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"실행 요약 저장: {summary_filename}")


def _process_category_poles(servers_data, output_base_dir, stats, category, max_normal=None, initial_count=0):
    """카테고리(break/normal)별 전주 데이터 수집. normal일 때 max_normal, initial_count 사용"""
    breakstate = 'B' if category == 'break' else 'N'
    cat_name = '파단' if category == 'break' else '정상'
    current_count = [initial_count]

    for server, server_info in servers_data.items():
        if server == "jt":
            continue
        if max_normal is not None and current_count[0] >= max_normal:
            break

        print(f"\n[{SERVERS.get(server, server)}] ({cat_name}) 전주 처리 시작")

        try:
            PDB.poledb_init(server)
            projects_data = server_info.get('projects', {})

            for project_idx, (project_name, project_info) in enumerate(projects_data.items(), 1):
                if max_normal is not None and current_count[0] >= max_normal:
                    break
                pole_ids = project_info.get('pole_ids', [])
                saved_count = count_saved_poles_in_project(output_base_dir, project_name, category)
                db_count = count_db_poles_in_project(server, project_name, breakstate)

                if saved_count == db_count:
                    continue

                saved_ids = _get_saved_pole_ids(output_base_dir, project_name, category)
                unsaved_ids = [pid for pid in pole_ids if pid not in saved_ids]
                if not unsaved_ids:
                    continue

                for poleid in tqdm(unsaved_ids, desc=f"  {project_name} ({cat_name} 수집)"):
                    if max_normal is not None and current_count[0] >= max_normal:
                        break
                    if category == 'break':
                        stats['total_poles'] += 1

                    try:
                        anal_result = get_pole_anal2_result(server, project_name, poleid)
                        if anal_result is None:
                            stats['skipped_poles'] += 1
                            continue
                        if anal_result['breakstate'] != breakstate:
                            continue

                        result = save_pole_raw_data(server, project_name, poleid, anal_result, output_base_dir)
                        if result is None:
                            stats['skipped_existing'] += 1
                        elif result:
                            stats['break_poles' if category == 'break' else 'normal_poles'] += 1
                            if category == 'normal':
                                current_count[0] += 1
                        else:
                            stats['error_poles'] += 1
                    except Exception as e:
                        print(f"    [{poleid}] 처리 오류: {e}")
                        stats['error_poles'] += 1

        except Exception as e:
            print(f"[{SERVERS.get(server, server)}] 서버 처리 오류: {e}")


def process_all_poles_from_json(json_file_path):
    """JSON에서 전주 목록을 읽어 원본 데이터 조회 및 저장"""
    print(f"\nJSON 파일 읽기: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_base_dir = os.path.join(current_dir, OUTPUT_DIR)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 통계 정보
    stats = {
        'total_poles': 0,
        'break_poles': 0,
        'normal_poles': 0,
        'skipped_poles': 0,
        'skipped_existing': 0,
        'skipped_limit': 0,
        'error_poles': 0,
    }
    servers_data = data.get('servers', {})

    # 1단계: 파단(B) 전주 수집
    print("\n1단계: 파단(B) 전주 데이터 수집 시작")
    _process_category_poles(servers_data, output_base_dir, stats, 'break')

    total_saved_break = 0
    total_saved_normal = 0
    for server, server_info in servers_data.items():
        if server == "jt":
            continue
        for project_name in server_info.get('projects', {}):
            total_saved_break += count_saved_poles_in_project(output_base_dir, project_name, 'break')
            total_saved_normal += count_saved_poles_in_project(output_base_dir, project_name, 'normal')
    max_normal_poles = total_saved_break * NORMAL_POLE_RATIO

    print(f"\n2단계: 정상(N) 전주 데이터 수집 시작")
    print(f"파단: {total_saved_break}개, 정상: {total_saved_normal}개, 목표: {max_normal_poles}개")

    if total_saved_break > 0 and total_saved_normal >= max_normal_poles:
        _save_run_summary_json(output_base_dir, json_file_path, stats, max_normal_poles, total_saved_break, total_saved_normal)
        return

    # 2단계: 정상(N) 전주 수집 (최대 파단의 NORMAL_POLE_RATIO배까지)
    _process_category_poles(
        servers_data, output_base_dir, stats, 'normal',
        max_normal=max_normal_poles, initial_count=total_saved_normal
    )

    # 최종 통계 출력
    print(f"\n전체 처리 완료")
    print(f"파단: {stats['break_poles']}개, 정상: {stats['normal_poles']}개")
    print(f"저장 위치: {output_base_dir}")

    # 실행 요약 통합 JSON 1개 저장 (1·2번 스크립트 저장 방식에 맞춤)
    _save_run_summary_json(
        output_base_dir, json_file_path, stats, max_normal_poles,
        total_saved_break, total_saved_normal + stats.get("normal_poles", 0),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("원본 전주 데이터 수집 시작")
    print("=" * 60)
    try:
        json_file = find_latest_json_file()
        process_all_poles_from_json(json_file)
        print("\n" + "=" * 60)
        print("원본 전주 데이터 수집 완료")
        print("=" * 60)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()

