#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4. merge_data 디렉토리의 병합 데이터 정보 확인 및 통계 시각화"""

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
merge_data_dir = Path(current_dir) / "4. merge_data"

# 각도 구간 (0~90, 90~180, 180~270, 270~360)
DEGREE_RANGES = [(0, 90), (90, 180), (180, 270), (270, 360)]
DEGREE_LABELS = ['0~90°', '90~180°', '180~270°', '270~360°']

# 높이 구간: 0~2m은 0.1m 단위, 2m 이상은 한 구간으로
# 부동소수점 오차 방지: i/10 형태로 정확한 경계 생성 (0.3이 [0.2,0.3)에 배치되는 버그 방지)
HEIGHT_BINS_01 = np.array([i / 10.0 for i in range(21)])  # 0, 0.1, ..., 2.0 (정확한 값)
HEIGHT_BIN_LABELS_01 = [f'{i/10:.1f}~{(i+1)/10:.1f}m' for i in range(20)]
HEIGHT_BINS = np.concatenate([HEIGHT_BINS_01, [100.0]])  # 2m 이상 포함
HEIGHT_BIN_LABELS = HEIGHT_BIN_LABELS_01 + ['2.0m 이상']


def _histogram_height(heights):
    """0~2m: 0.1m 단위(20구간), 2m 이상: 1구간. (hist, labels) 반환"""
    if not heights:
        return np.zeros(len(HEIGHT_BIN_LABELS)), HEIGHT_BIN_LABELS
    arr = np.round(np.array(heights, dtype=float), 2)  # 부동소수점 오차 방지
    hist, _ = np.histogram(arr, bins=HEIGHT_BINS)
    return hist, HEIGHT_BIN_LABELS


def _histogram_height_by_files(file_heights_list):
    """파일별 높이 리스트를 받아, 각 구간에 데이터가 있는 파일 수로 집계. (hist, labels) 반환"""
    hist = np.zeros(len(HEIGHT_BINS) - 1)
    for file_heights in file_heights_list:
        if not file_heights:
            continue
        # 부동소수점 오차 방지: 0.1m 단위로 반올림 후 구간 배치
        arr = np.round(np.array(file_heights, dtype=float), 2)
        bin_indices = np.digitize(arr, HEIGHT_BINS, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        for idx in np.unique(bin_indices):
            hist[idx] += 1
    return hist, HEIGHT_BIN_LABELS


def _degree_to_quadrant(deg):
    """각도를 구간 인덱스(0~3)로 변환"""
    if 0 <= deg < 90:
        return 0
    elif 90 <= deg < 180:
        return 1
    elif 180 <= deg < 270:
        return 2
    elif 270 <= deg < 360:
        return 3
    return -1


def collect_merge_data_stats():
    """merge_data 디렉토리에서 통계 수집"""
    if not merge_data_dir.exists():
        print(f"오류: 디렉토리를 찾을 수 없습니다: {merge_data_dir}")
        return None

    stats = {
        'break': {'files': 0, 'poles': 0, 'projects': set(), 'degree_rows': [0, 0, 0, 0], 'degree_files': [0, 0, 0, 0], 'file_heights': [], 'break_heights': []},
        'normal': {'files': 0, 'poles': 0, 'projects': set(), 'degree_rows': [0, 0, 0, 0], 'degree_files': [0, 0, 0, 0], 'file_heights': []},
    }

    # 파단 높이(breakheight) 수집: *_OUT_processed_break_info.json
    break_data_path = merge_data_dir / "break"
    if break_data_path.exists():
        for info_file in break_data_path.rglob("*_OUT_processed_break_info.json"):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                bh = info.get('breakheight')
                if bh is not None:
                    try:
                        stats['break']['break_heights'].append(float(bh))
                    except (TypeError, ValueError):
                        pass
            except Exception:
                continue

    # 먼저 모든 병합 CSV 경로 수집
    all_csv_paths = []
    for data_type in ['break', 'normal']:
        data_path = merge_data_dir / data_type
        if not data_path.exists():
            continue
        for project_dir in data_path.iterdir():
            if not project_dir.is_dir():
                continue
            stats[data_type]['projects'].add(project_dir.name)
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                stats[data_type]['poles'] += 1
                for csv_path in pole_dir.glob("*_OUT_processed.csv"):
                    all_csv_paths.append((data_type, csv_path))

    print(f"\n병합 CSV 파일 분석 중... (총 {len(all_csv_paths):,}개)")
    for data_type, csv_path in tqdm(all_csv_paths, desc="  CSV 분석", unit="파일"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'degree' not in df.columns or 'height' not in df.columns:
                continue

            stats[data_type]['files'] += 1

            # 각도 구간별 집계
            file_quadrants = set()
            for deg in df['degree'].dropna():
                q = _degree_to_quadrant(float(deg))
                if q >= 0:
                    stats[data_type]['degree_rows'][q] += 1
                    file_quadrants.add(q)
            for q in file_quadrants:
                stats[data_type]['degree_files'][q] += 1

            # 높이 수집 (파일별 - 파일 개수 집계용)
            file_heights = df['height'].dropna().astype(float).tolist()
            stats[data_type]['file_heights'].append(file_heights)

        except Exception:
            continue

    return stats


def analyze_merge_data():
    """merge_data 디렉토리 분석 및 통계 반환"""
    print("=" * 80)
    print("4. merge_data 디렉토리 정보 분석")
    print("=" * 80)

    stats = collect_merge_data_stats()
    if stats is None:
        return None

    print("\n[파단]")
    print(f"  프로젝트: {len(stats['break']['projects'])}개, 전주: {stats['break']['poles']}개, 병합 CSV: {stats['break']['files']}개")
    print(f"  파단 높이(breakheight) 레코드: {len(stats['break'].get('break_heights', []))}개")
    print("\n[정상]")
    print(f"  프로젝트: {len(stats['normal']['projects'])}개, 전주: {stats['normal']['poles']}개, 병합 CSV: {stats['normal']['files']}개")

    total_files = stats['break']['files'] + stats['normal']['files']
    print(f"\n[전체] 병합된 CSV 파일: {total_files}개")
    return stats


def plot_merge_data_stats(stats):
    """막대 그래프로 통계 시각화 (높이 0.1m 단위)"""
    if stats is None:
        return

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    colors_break = '#e74c3c'
    colors_normal = '#2ecc71'

    # 1. 병합된 파일 수 (파단 vs 정상)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['파단', '정상', '전체']
    values = [stats['break']['files'], stats['normal']['files'], stats['break']['files'] + stats['normal']['files']]
    bar_colors = [colors_break, colors_normal, '#3498db']
    bars = ax1.bar(labels, values, color=bar_colors)
    ax1.set_title('병합된 CSV 파일 수')
    ax1.set_ylabel('개수')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:,}', ha='center', va='bottom', fontsize=11)

    # 2. 각도 구간별 데이터 행 수 비율 (파단)
    ax2 = fig.add_subplot(gs[0, 1])
    deg_rows_break = stats['break']['degree_rows']
    total_break = sum(deg_rows_break)
    if total_break > 0:
        ratios_break = [r / total_break * 100 for r in deg_rows_break]
    else:
        ratios_break = [0, 0, 0, 0]
    bars2 = ax2.bar(DEGREE_LABELS, ratios_break, color=colors_break, alpha=0.8)
    ax2.set_title('각도 구간별 데이터 비율 (파단)')
    ax2.set_ylabel('비율 (%)')
    for bar, val, cnt in zip(bars2, ratios_break, deg_rows_break):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%\n({cnt:,})', ha='center', va='bottom', fontsize=9)

    # 3. 각도 구간별 데이터 행 수 비율 (정상)
    ax3 = fig.add_subplot(gs[0, 2])
    deg_rows_normal = stats['normal']['degree_rows']
    total_normal = sum(deg_rows_normal)
    if total_normal > 0:
        ratios_normal = [r / total_normal * 100 for r in deg_rows_normal]
    else:
        ratios_normal = [0, 0, 0, 0]
    bars3 = ax3.bar(DEGREE_LABELS, ratios_normal, color=colors_normal, alpha=0.8)
    ax3.set_title('각도 구간별 데이터 비율 (정상)')
    ax3.set_ylabel('비율 (%)')
    for bar, val, cnt in zip(bars3, ratios_normal, deg_rows_normal):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%\n({cnt:,})', ha='center', va='bottom', fontsize=9)

    # 4. 각도 구간별 파일 수 비율 (전체)
    ax4 = fig.add_subplot(gs[1, 0])
    deg_files_break = stats['break']['degree_files']
    deg_files_normal = stats['normal']['degree_files']
    deg_files_total = [a + b for a, b in zip(deg_files_break, deg_files_normal)]
    total_deg_files = sum(deg_files_total)
    if total_deg_files > 0:
        file_ratios = [f / total_deg_files * 100 for f in deg_files_total]
    else:
        file_ratios = [0, 0, 0, 0]
    x = np.arange(len(DEGREE_LABELS))
    width = 0.35
    bars_b = ax4.bar(x - width/2, deg_files_break, width, label='파단', color=colors_break)
    bars_n = ax4.bar(x + width/2, deg_files_normal, width, label='정상', color=colors_normal)
    ax4.set_title('각도 구간별 해당 파일 수')
    ax4.set_ylabel('파일 수')
    ax4.set_xticks(x)
    ax4.set_xticklabels(DEGREE_LABELS)
    ax4.legend()

    # 5. 높이 분포 (파단, 0~2m 0.1단위, 2m 이상) - 파일 수
    ax5 = fig.add_subplot(gs[1, 1])
    file_heights_break = stats['break'].get('file_heights', [])
    if file_heights_break:
        hist_b, bin_labels = _histogram_height_by_files(file_heights_break)
        ax5.bar(range(len(hist_b)), hist_b, color=colors_break, alpha=0.8)
        ax5.set_xticks(range(len(bin_labels)))
        ax5.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax5.set_title(f'높이 분포 (파단, 0~2m 0.1단위·2m이상, 파일 {len(file_heights_break):,}개)')
        ax5.set_ylabel('파일 수')
    else:
        ax5.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('높이 분포 (파단)')

    # 6. 높이 분포 (정상, 0~2m 0.1단위, 2m 이상) - 파일 수
    ax6 = fig.add_subplot(gs[1, 2])
    file_heights_normal = stats['normal'].get('file_heights', [])
    if file_heights_normal:
        hist_n, bin_labels = _histogram_height_by_files(file_heights_normal)
        ax6.bar(range(len(hist_n)), hist_n, color=colors_normal, alpha=0.8)
        ax6.set_xticks(range(len(bin_labels)))
        ax6.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax6.set_title(f'높이 분포 (정상, 0~2m 0.1단위·2m이상, 파일 {len(file_heights_normal):,}개)')
        ax6.set_ylabel('파일 수')
    else:
        ax6.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('높이 분포 (정상)')

    # 7. 파단 높이 통계 (breakheight, 0~2m 0.1단위, 2m 이상) - 파일 수
    ax7 = fig.add_subplot(gs[2, :])
    break_heights = stats['break'].get('break_heights', [])
    if break_heights:
        hist_break, bin_labels = _histogram_height(break_heights)  # breakheight는 파일당 1개
        ax7.bar(range(len(hist_break)), hist_break, color=colors_break, alpha=0.8)
        ax7.set_xticks(range(len(bin_labels)))
        ax7.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax7.set_title(f'파단 높이 통계 (breakheight, 0~2m 0.1단위·2m이상, 파일 {len(break_heights):,}개)')
        ax7.set_ylabel('파일 수')
        ax7.set_xlabel('높이 구간 (m)')
    else:
        ax7.text(0.5, 0.5, '파단 높이 데이터 없음', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('파단 높이 통계 (breakheight)')

    fig.suptitle('4. merge_data 통계 (높이: 0~2m 0.1단위, 2m 이상)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_summary_table(stats):
    """요약 테이블 출력"""
    if stats is None:
        return

    print("\n" + "=" * 90)
    print("요약 테이블")
    print("=" * 90)
    print(f"{'구분':<10} {'프로젝트':>10} {'전주수':>10} {'병합CSV':>12} {'0~90°':>12} {'90~180°':>12} {'180~270°':>12} {'270~360°':>12}")
    print("-" * 90)

    for dtype, label in [('break', '파단'), ('normal', '정상')]:
        s = stats[dtype]
        total_rows = sum(s['degree_rows'])
        if total_rows > 0:
            r0 = s['degree_rows'][0] / total_rows * 100
            r1 = s['degree_rows'][1] / total_rows * 100
            r2 = s['degree_rows'][2] / total_rows * 100
            r3 = s['degree_rows'][3] / total_rows * 100
        else:
            r0 = r1 = r2 = r3 = 0
        print(f"{label:<10} {len(s['projects']):>10} {s['poles']:>10} {s['files']:>12} {r0:>11.1f}% {r1:>11.1f}% {r2:>11.1f}% {r3:>11.1f}%")

    print("=" * 90)


if __name__ == "__main__":
    stats = analyze_merge_data()
    print_summary_table(stats)
    plot_merge_data_stats(stats)
