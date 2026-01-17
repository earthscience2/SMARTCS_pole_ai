#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4. merge_data 디렉토리의 정보를 확인하는 스크립트
- 프로젝트 수
- 파일 수 (CSV, JSON, PNG)
- 파단 전주 수
- 정상 전주 수
- 기타 통계 정보
"""

import os
import json
from pathlib import Path

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
merge_data_dir = Path(current_dir) / "4. merge_data"


def analyze_merge_data():
    """merge_data 디렉토리 분석"""
    if not merge_data_dir.exists():
        print(f"오류: 디렉토리를 찾을 수 없습니다: {merge_data_dir}")
        return
    
    print("=" * 80)
    print("4. merge_data 디렉토리 정보 분석")
    print("=" * 80)
    print()
    
    # 전체 통계
    total_stats = {
        'break': {
            'projects': set(),
            'poles': 0,
            'csv_files': 0,
            'json_files': 0,
            'png_files': 0,
            'other_files': 0
        },
        'normal': {
            'projects': set(),
            'poles': 0,
            'csv_files': 0,
            'json_files': 0,
            'png_files': 0,
            'other_files': 0
        }
    }
    
    # break와 normal 디렉토리 처리
    for data_type in ['break', 'normal']:
        data_type_path = merge_data_dir / data_type
        
        if not data_type_path.exists():
            print(f"[{data_type.upper()}] 디렉토리가 존재하지 않습니다.")
            continue
        
        print(f"[{data_type.upper()}] 데이터 분석 중...")
        
        # 프로젝트 디렉토리 찾기
        projects = [d for d in data_type_path.iterdir() if d.is_dir()]
        
        for project_dir in projects:
            project_name = project_dir.name
            total_stats[data_type]['projects'].add(project_name)
            
            # 전주 디렉토리 찾기
            pole_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
            
            for pole_dir in pole_dirs:
                poleid = pole_dir.name
                total_stats[data_type]['poles'] += 1
                
                # 전주 디렉토리 내 파일 카운트
                csv_files = list(pole_dir.glob("*.csv"))
                json_files = list(pole_dir.glob("*.json"))
                png_files = list(pole_dir.glob("*.png"))
                
                csv_count = len(csv_files)
                json_count = len(json_files)
                png_count = len(png_files)
                
                total_stats[data_type]['csv_files'] += csv_count
                total_stats[data_type]['json_files'] += json_count
                total_stats[data_type]['png_files'] += png_count
    
    # 결과 출력
    print()
    print("=" * 80)
    print("전체 통계")
    print("=" * 80)
    print()
    
    # 전체 프로젝트 수 (break와 normal의 합집합)
    all_projects = total_stats['break']['projects'] | total_stats['normal']['projects']
    print(f"전체 프로젝트 수: {len(all_projects)}개")
    print(f"  - 파단 데이터 프로젝트: {len(total_stats['break']['projects'])}개")
    print(f"  - 정상 데이터 프로젝트: {len(total_stats['normal']['projects'])}개")
    print()
    
    # 파단 전주 통계
    print(f"[파단 전주]")
    print(f"  전주 수: {total_stats['break']['poles']}개")
    print(f"  CSV 파일 수: {total_stats['break']['csv_files']:,}개")
    print(f"  JSON 파일 수: {total_stats['break']['json_files']:,}개")
    print(f"  PNG 파일 수: {total_stats['break']['png_files']:,}개")
    if total_stats['break']['poles'] > 0:
        print(f"  전주당 평균 CSV 파일: {total_stats['break']['csv_files'] / total_stats['break']['poles']:.1f}개")
        print(f"  전주당 평균 JSON 파일: {total_stats['break']['json_files'] / total_stats['break']['poles']:.1f}개")
        print(f"  전주당 평균 PNG 파일: {total_stats['break']['png_files'] / total_stats['break']['poles']:.1f}개")
    print()
    
    # 정상 전주 통계
    print(f"[정상 전주]")
    print(f"  전주 수: {total_stats['normal']['poles']}개")
    print(f"  CSV 파일 수: {total_stats['normal']['csv_files']:,}개")
    print(f"  JSON 파일 수: {total_stats['normal']['json_files']:,}개")
    print(f"  PNG 파일 수: {total_stats['normal']['png_files']:,}개")
    if total_stats['normal']['poles'] > 0:
        print(f"  전주당 평균 CSV 파일: {total_stats['normal']['csv_files'] / total_stats['normal']['poles']:.1f}개")
        print(f"  전주당 평균 JSON 파일: {total_stats['normal']['json_files'] / total_stats['normal']['poles']:.1f}개")
        print(f"  전주당 평균 PNG 파일: {total_stats['normal']['png_files'] / total_stats['normal']['poles']:.1f}개")
    print()
    
    # 전체 합계
    total_poles = total_stats['break']['poles'] + total_stats['normal']['poles']
    total_csv = total_stats['break']['csv_files'] + total_stats['normal']['csv_files']
    total_json = total_stats['break']['json_files'] + total_stats['normal']['json_files']
    total_png = total_stats['break']['png_files'] + total_stats['normal']['png_files']
    
    print(f"[전체 합계]")
    print(f"  전체 전주 수: {total_poles}개")
    print(f"  전체 CSV 파일 수: {total_csv:,}개")
    print(f"  전체 JSON 파일 수: {total_json:,}개")
    print(f"  전체 PNG 파일 수: {total_png:,}개")
    print()
    
    # 요약 테이블
    print("=" * 80)
    print("요약 테이블")
    print("=" * 80)
    print()
    print(f"{'구분':<15} {'프로젝트 수':<15} {'전주 수':<15} {'CSV 파일 수':<20} {'JSON 파일 수':<20} {'PNG 파일 수':<20}")
    print("-" * 100)
    print(f"{'파단':<15} {len(total_stats['break']['projects']):<15} {total_stats['break']['poles']:<15} {total_stats['break']['csv_files']:<20,} {total_stats['break']['json_files']:<20,} {total_stats['break']['png_files']:<20,}")
    print(f"{'정상':<15} {len(total_stats['normal']['projects']):<15} {total_stats['normal']['poles']:<15} {total_stats['normal']['csv_files']:<20,} {total_stats['normal']['json_files']:<20,} {total_stats['normal']['png_files']:<20,}")
    print(f"{'전체':<15} {len(all_projects):<15} {total_poles:<15} {total_csv:<20,} {total_json:<20,} {total_png:<20,}")
    print()
    
    print("=" * 80)
    print("분석 완료")
    print("=" * 80)


if __name__ == "__main__":
    analyze_merge_data()
