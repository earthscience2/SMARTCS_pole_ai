#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메인 서버 연결 테스트 스크립트
"""

import sys
import os
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import poledb as PDB
from config import poleconf

def test_server_connection(server_name):
    """
    서버 연결 테스트
    
    Args:
        server_name: 서버 이름 ('main', 'is', 'kh', 'jt')
    """
    print("=" * 60)
    print(f"[{server_name.upper()} 서버] 연결 테스트 시작")
    print("=" * 60)
    
    # 서버별 호스트 정보 출력 (JT 서버는 테스트 대상에서 제외)
    server_hosts = {
        'main': '210.105.85.3:3306',
        'is': 'smartpole-is.iptime.org:33306',
        'kh': 'smartpole-kh.iptime.org:33306',
    }
    
    if server_name in server_hosts:
        host = server_hosts[server_name]
        print(f"서버 호스트: {host}")
    else:
        print(f"경고: 알 수 없는 서버 이름: {server_name}")
        return False

    # poleconf.db() 함수에서 DB 설정값을 가져옴
    try:
        _, dbname, user, _ = poleconf.db(server_name)
        print(f"데이터베이스: {dbname}")
        print(f"사용자: {user}")
    except Exception as e:
        print(f"DB 설정 조회 중 오류: {e}")
        return False
    print("-" * 60)
    
    # 1. 연결 시도
    print("\n[1단계] 데이터베이스 연결 시도...")
    try:
        PDB.poledb_init(server_name)
        print("✓ 연결 시도 완료")
    except Exception as e:
        print(f"✗ 연결 실패: {e}")
        print(f"  에러 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 연결 상태 확인
    print("\n[2단계] 연결 상태 확인...")
    if PDB.poledb_conn is None:
        print("✗ poledb_conn이 None입니다.")
        return False
    else:
        print("✓ poledb_conn 객체 생성됨")
    
    # 3. Ping 테스트 (간단한 쿼리 실행)
    print("\n[3단계] Ping 테스트 (SELECT NOW() 실행)...")
    try:
        PDB.ping()
        print("✓ Ping 성공")
    except Exception as e:
        print(f"✗ Ping 실패: {e}")
        print(f"  에러 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 간단한 쿼리 테스트
    print("\n[4단계] 간단한 쿼리 테스트...")
    try:
        query = "SELECT COUNT(*) as count FROM tb_anal_state LIMIT 1"
        result = PDB.poledb_conn.do_select_pd(query, [])
        if result is not None and not result.empty:
            count = result.iloc[0]['count']
            print(f"✓ 쿼리 성공: tb_anal_state 테이블에 {count}개 행 존재")
        else:
            print("⚠ 쿼리는 성공했지만 결과가 없습니다.")
    except Exception as e:
        print(f"✗ 쿼리 실패: {e}")
        print(f"  에러 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 프로젝트 목록 조회 테스트
    print("\n[5단계] 프로젝트 목록 조회 테스트...")
    try:
        project_list = PDB.groupname_info()
        if project_list is not None and len(project_list) > 0:
            print(f"✓ 프로젝트 목록 조회 성공: {len(project_list)}개 프로젝트")
            print(f"  처음 5개 프로젝트: {project_list[:5]}")
        else:
            print("⚠ 프로젝트 목록이 비어있습니다.")
    except Exception as e:
        print(f"✗ 프로젝트 목록 조회 실패: {e}")
        print(f"  에러 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print(f"[{server_name.upper()} 서버] 모든 테스트 통과! ✓")
    print("=" * 60)
    return True

def test_all_servers():
    """
    모든 서버 연결 테스트
    """
    # JT 서버는 테스트 대상에서 제외
    servers = ['main', 'is', 'kh']
    results = {}
    
    print("\n" + "=" * 60)
    print("전체 서버 연결 테스트")
    print("=" * 60)
    
    for server in servers:
        try:
            result = test_server_connection(server)
            results[server] = result
            print("\n")
        except KeyboardInterrupt:
            print("\n\n사용자에 의해 중단되었습니다.")
            break
        except Exception as e:
            print(f"\n✗ [{server}] 예상치 못한 오류: {e}")
            results[server] = False
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for server, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{server:10s}: {status}")
    
    return results

if __name__ == "__main__":
    print("메인 서버 연결 테스트 스크립트")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(sys.argv) > 1:
        # 특정 서버만 테스트
        server_name = sys.argv[1]
        test_server_connection(server_name)
    else:
        # 모든 서버 테스트
        test_all_servers()
