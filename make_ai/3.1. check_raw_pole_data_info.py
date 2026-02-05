#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3. raw_pole_data 디렉토리의 정보를 확인하고 통계 시각화"""

import os
import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = Path(current_dir) / "3. raw_pole_data"
project_info_dir = Path(current_dir) / "1. project_info_list"
anal_pole_dir = Path(current_dir) / "2. anal_pole_list"

SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}


def load_stats_from_json_files():
    """
    project_list_all_*.json, anal2_poles_all_*.json에서 서버별 통계 계산
    프로젝트수, 전주수, 파단전주수, 정상전주수는 JSON 기준 (실제 수집 여부와 무관)
    """
    server_stats = {}
    for server in SERVERS.keys():
        server_stats[server] = {
            'projects': set(),
            'projects_count': 0,
            'poles_break': 0,
            'poles_normal': 0,
            'csv_files': 0,
            'out_files': 0,
            'in_files': 0,
        }
    
    # anal2_poles_all_*.json 로드 (프로젝트수, 전주수, 파단/정상 전주수)
    anal_pattern = str(anal_pole_dir / "anal2_poles_all_*.json")
    anal_files = glob.glob(anal_pattern)
    if not anal_files:
        print(f"오류: anal2_poles_all JSON을 찾을 수 없습니다: {anal_pattern}")
        return None
    
    anal_file = max(anal_files, key=os.path.getmtime)
    print(f"분석 전주 목록 로드: {anal_file}")
    
    with open(anal_file, 'r', encoding='utf-8') as f:
        anal_data = json.load(f)
    
    servers_data = anal_data.get('servers', {})
    for server in SERVERS.keys():
        if server not in servers_data:
            continue
        server_info = servers_data[server]
        projects = server_info.get('projects', {})
        
        if isinstance(projects, list):
            # 리스트 형태 (이전 형식) - 사용 안 함, dict 기대
            continue
        
        server_stats[server]['projects_count'] = len(projects)
        server_stats[server]['projects'] = set(projects.keys())
        
        for proj_name, proj_info in projects.items():
            if isinstance(proj_info, dict):
                server_stats[server]['poles_break'] += proj_info.get('break_count', 0)
                server_stats[server]['poles_normal'] += proj_info.get('normal_count', 0)
    
    # project_list_all_*.json에서 프로젝트 매핑 확인 (anal2에 없을 수 있는 프로젝트)
    proj_pattern = str(project_info_dir / "project_list_all_*.json")
    proj_files = glob.glob(proj_pattern)
    if proj_files:
        proj_file = max(proj_files, key=os.path.getmtime)
        print(f"프로젝트 목록 로드: {proj_file}")
    
    return server_stats, servers_data


def load_file_counts_from_raw_data(project_to_server):
    """raw_pole_data에서 CSV/OUT/IN 파일 수 집계 (실제 수집된 데이터 기준)"""
    file_stats = {s: {'csv_files': 0, 'out_files': 0, 'in_files': 0} for s in SERVERS.keys()}
    file_stats['unknown'] = {'csv_files': 0, 'out_files': 0, 'in_files': 0}
    
    if not raw_data_dir.exists():
        return file_stats
    
    for data_type in ['break', 'normal']:
        data_type_path = raw_data_dir / data_type
        if not data_type_path.exists():
            continue
        
        for project_dir in data_type_path.iterdir():
            if not project_dir.is_dir():
                continue
            project_name = project_dir.name
            server = project_to_server.get(project_name, 'unknown')
            
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                for csv_file in pole_dir.glob("*.csv"):
                    file_stats[server]['csv_files'] += 1
                    filename = csv_file.name.upper()
                    if '_OUT_' in filename:
                        file_stats[server]['out_files'] += 1
                    elif '_IN_' in filename:
                        file_stats[server]['in_files'] += 1
    
    return file_stats


def build_project_to_server_mapping(servers_data):
    """anal2_poles_all의 servers 구조에서 프로젝트->서버 매핑 생성"""
    project_to_server = {}
    for server, server_info in servers_data.items():
        projects = server_info.get('projects', {})
        if isinstance(projects, dict):
            for proj_name in projects.keys():
                project_to_server[proj_name] = server
    return project_to_server


def analyze_raw_pole_data_by_server():
    """JSON 파일 기준 통계 + raw_pole_data 파일 수 통합"""
    print("=" * 80)
    print("서버별 통계 (project_list_all, anal2_poles_all JSON 기준)")
    print("=" * 80)
    
    result = load_stats_from_json_files()
    if result is None:
        return None
    
    server_stats, servers_data = result
    
    # 프로젝트->서버 매핑 (raw_pole_data 파일 집계용)
    project_to_server = build_project_to_server_mapping(servers_data)
    
    # raw_pole_data에서 CSV/OUT/IN 파일 수 추가
    file_stats = load_file_counts_from_raw_data(project_to_server)
    for server in SERVERS.keys():
        server_stats[server]['csv_files'] = file_stats.get(server, {}).get('csv_files', 0)
        server_stats[server]['out_files'] = file_stats.get(server, {}).get('out_files', 0)
        server_stats[server]['in_files'] = file_stats.get(server, {}).get('in_files', 0)
    
    return server_stats


def plot_server_stats(server_stats):
    """서버별 통계를 matplotlib으로 시각화 (저장 없이 show만, 프로젝트/전주/파단/정상: JSON 기준)"""
    if server_stats is None:
        return
    
    servers_to_show = [s for s in SERVERS.keys() if s in server_stats]
    server_labels = [SERVERS[s] for s in servers_to_show]
    
    # 통계 데이터 추출 (프로젝트수/전주수/파단/정상: anal2_poles_all JSON 기준)
    projects_count = [server_stats[s].get('projects_count', len(server_stats[s].get('projects', []))) for s in servers_to_show]
    poles_total = [server_stats[s]['poles_break'] + server_stats[s]['poles_normal'] for s in servers_to_show]
    csv_count = [server_stats[s]['csv_files'] for s in servers_to_show]
    out_count = [server_stats[s]['out_files'] for s in servers_to_show]
    in_count = [server_stats[s]['in_files'] for s in servers_to_show]
    break_count = [server_stats[s]['poles_break'] for s in servers_to_show]
    normal_count = [server_stats[s]['poles_normal'] for s in servers_to_show]
    
    # Figure 생성: 상단 2x3 (기본 통계), 하단 1x3 (수집된 파일 정보)
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.3)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # 파랑, 빨강, 초록
    
    def _add_bar_values(ax, bars, values):
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:,}' if isinstance(val, int) else str(val), ha='center', va='bottom', fontsize=10)
    
    # === 상단: 프로젝트·전주·파단·정상 통계 ===
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(server_labels, projects_count, color=colors)
    ax1.set_title('프로젝트 수')
    ax1.set_ylabel('개수')
    _add_bar_values(ax1, bars1, projects_count)
    
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(server_labels, poles_total, color=colors)
    ax2.set_title('전주 수 (전체)')
    ax2.set_ylabel('개수')
    _add_bar_values(ax2, bars2, poles_total)
    
    ax3 = fig.add_subplot(gs[0, 2])
    x = range(len(server_labels))
    width = 0.6
    ax3.bar(x, break_count, width, label='파단', color='#e74c3c')
    ax3.bar(x, normal_count, width, bottom=break_count, label='정상', color='#2ecc71')
    ax3.set_title('파단/정상 비교')
    ax3.set_ylabel('개수')
    ax3.set_xticks(x)
    ax3.set_xticklabels(server_labels)
    ax3.legend()
    for i, (b, n) in enumerate(zip(break_count, normal_count)):
        ax3.text(i, b + n + 0.5, f'{b+n:,}', ha='center', va='bottom', fontsize=10)
    
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(server_labels, break_count, color='#e74c3c')
    ax4.set_title('파단 전주 수')
    ax4.set_ylabel('개수')
    _add_bar_values(ax4, bars4, break_count)
    
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(server_labels, normal_count, color='#2ecc71')
    ax5.set_title('정상 전주 수')
    ax5.set_ylabel('개수')
    _add_bar_values(ax5, bars5, normal_count)
    
    # gs[1, 2] 비움
    
    # === 하단: 수집된 파일 정보 (CSV, OUT, IN) ===
    ax6 = fig.add_subplot(gs[2, 0])
    bars6 = ax6.bar(server_labels, csv_count, color=colors)
    ax6.set_title('CSV 파일 수 (수집)')
    ax6.set_ylabel('개수')
    _add_bar_values(ax6, bars6, csv_count)
    
    ax7 = fig.add_subplot(gs[2, 1])
    bars7 = ax7.bar(server_labels, out_count, color=colors)
    ax7.set_title('OUT 파일 수 (수집)')
    ax7.set_ylabel('개수')
    _add_bar_values(ax7, bars7, out_count)
    
    ax8 = fig.add_subplot(gs[2, 2])
    bars8 = ax8.bar(server_labels, in_count, color=colors)
    ax8.set_title('IN 파일 수 (수집)')
    ax8.set_ylabel('개수')
    _add_bar_values(ax8, bars8, in_count)
    
    fig.suptitle('서버별 통계 (상단: JSON 기준 / 하단: 수집된 파일 정보)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_server_stats(server_stats):
    """서버별 통계 콘솔 출력 (프로젝트수/전주수/파단/정상: JSON 기준)"""
    if server_stats is None:
        return
    
    print("\n" + "=" * 110)
    print("서버별 통계 요약 (프로젝트·전주·파단·정상: anal2_poles_all JSON 기준 / CSV·OUT·IN: raw_pole_data 수집 기준)")
    print("=" * 110)
    print(f"{'서버':<12} {'프로젝트수':>10} {'전주수':>10} {'CSV파일':>12} {'OUT파일':>12} {'IN파일':>12} {'파단전주':>10} {'정상전주':>10}")
    print("-" * 110)
    
    total = {'projects': 0, 'poles': 0, 'csv': 0, 'out': 0, 'in': 0, 'break': 0, 'normal': 0}
    
    for server in SERVERS.keys():
        if server not in server_stats:
            continue
        s = server_stats[server]
        server_name = SERVERS[server]
        poles = s['poles_break'] + s['poles_normal']
        proj_count = s.get('projects_count', len(s.get('projects', [])))
        
        print(f"{server_name:<12} {proj_count:>10} {poles:>10,} {s['csv_files']:>12,} {s['out_files']:>12,} {s['in_files']:>12,} {s['poles_break']:>10,} {s['poles_normal']:>10,}")
        
        total['projects'] += proj_count
        total['poles'] += poles
        total['csv'] += s['csv_files']
        total['out'] += s['out_files']
        total['in'] += s['in_files']
        total['break'] += s['poles_break']
        total['normal'] += s['poles_normal']
    
    print("-" * 110)
    print(f"{'전체':<12} {total['projects']:>10} {total['poles']:>10,} {total['csv']:>12,} {total['out']:>12,} {total['in']:>12,} {total['break']:>10,} {total['normal']:>10,}")
    print("=" * 110)


if __name__ == "__main__":
    server_stats = analyze_raw_pole_data_by_server()
    print_server_stats(server_stats)
    plot_server_stats(server_stats)
