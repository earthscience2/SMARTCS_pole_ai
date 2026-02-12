#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_data(4단계) 품질/분포 통계를 확인한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CURRENT_DIR = Path(__file__).resolve().parent
MERGE_DATA_DIR = CURRENT_DIR / "4. merge_data"

DEGREE_LABELS = ["0~90°", "90~180°", "180~270°", "270~360°"]
HEIGHT_BINS = np.concatenate([np.array([i / 10.0 for i in range(21)]), [100.0]])
HEIGHT_LABELS = [f"{i/10:.1f}~{(i+1)/10:.1f}m" for i in range(20)] + ["2.0m 이상"]


def degree_quadrant(deg: float) -> int:
    if 0 <= deg < 90:
        return 0
    if 90 <= deg < 180:
        return 1
    if 180 <= deg < 270:
        return 2
    if 270 <= deg < 360:
        return 3
    return -1


def histogram_file_coverage(file_heights: List[List[float]]) -> np.ndarray:
    """파일 단위 높이 커버리지 히스토그램."""
    hist = np.zeros(len(HEIGHT_BINS) - 1, dtype=np.int64)
    for heights in file_heights:
        if not heights:
            continue
        arr = np.round(np.asarray(heights, dtype=float), 2)
        idx = np.digitize(arr, HEIGHT_BINS, right=False) - 1
        idx = np.clip(idx, 0, len(hist) - 1)
        for one in np.unique(idx):
            hist[one] += 1
    return hist


def histogram_break_heights(break_heights: List[float]) -> np.ndarray:
    if not break_heights:
        return np.zeros(len(HEIGHT_BINS) - 1, dtype=np.int64)
    arr = np.round(np.asarray(break_heights, dtype=float), 2)
    hist, _ = np.histogram(arr, bins=HEIGHT_BINS)
    return hist


def collect_stats(data_dir: Path) -> Optional[Dict]:
    if not data_dir.exists():
        print(f"오류: 디렉터리가 없습니다. {data_dir}")
        return None

    stats = {
        "break": {
            "projects": set(),
            "poles": 0,
            "files": 0,
            "degree_rows": [0, 0, 0, 0],
            "degree_files": [0, 0, 0, 0],
            "file_heights": [],
            "break_heights": [],
        },
        "normal": {
            "projects": set(),
            "poles": 0,
            "files": 0,
            "degree_rows": [0, 0, 0, 0],
            "degree_files": [0, 0, 0, 0],
            "file_heights": [],
        },
    }

    break_dir = data_dir / "break"
    if break_dir.exists():
        for info_path in break_dir.rglob("*_OUT_processed_break_info.json"):
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                h = payload.get("breakheight")
                if h is not None:
                    stats["break"]["break_heights"].append(float(h))
            except Exception:
                continue

    all_csv: List[Tuple[str, Path]] = []
    for category in ("break", "normal"):
        cat_dir = data_dir / category
        if not cat_dir.exists():
            continue
        for project_dir in cat_dir.iterdir():
            if not project_dir.is_dir():
                continue
            stats[category]["projects"].add(project_dir.name)
            for pole_dir in project_dir.iterdir():
                if not pole_dir.is_dir():
                    continue
                stats[category]["poles"] += 1
                for csv_path in pole_dir.glob("*_OUT_processed.csv"):
                    all_csv.append((category, csv_path))

    print(f"CSV 분석 대상: {len(all_csv):,}개")
    for category, csv_path in tqdm(all_csv, desc="CSV 분석", unit="file"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "degree" not in df.columns or "height" not in df.columns:
                continue
            stats[category]["files"] += 1

            file_quads = set()
            for deg in df["degree"].dropna():
                q = degree_quadrant(float(deg))
                if q >= 0:
                    stats[category]["degree_rows"][q] += 1
                    file_quads.add(q)
            for q in file_quads:
                stats[category]["degree_files"][q] += 1

            heights = df["height"].dropna().astype(float).tolist()
            stats[category]["file_heights"].append(heights)
        except Exception:
            continue

    return stats


def print_summary(stats: Dict) -> None:
    print("\n" + "=" * 100)
    print("4. merge_data 요약")
    print("=" * 100)
    print(
        f"{'구분':<10} {'프로젝트':>10} {'전주':>10} {'CSV':>10} "
        f"{'0~90°':>12} {'90~180°':>12} {'180~270°':>12} {'270~360°':>12}"
    )
    print("-" * 100)

    for key, label in (("break", "파단"), ("normal", "정상")):
        s = stats[key]
        rows = s["degree_rows"]
        total_rows = sum(rows)
        ratios = [r / total_rows * 100 if total_rows else 0.0 for r in rows]
        print(
            f"{label:<10} {len(s['projects']):>10} {s['poles']:>10,} {s['files']:>10,} "
            f"{ratios[0]:>11.1f}% {ratios[1]:>11.1f}% {ratios[2]:>11.1f}% {ratios[3]:>11.1f}%"
        )
    print("=" * 100)


def plot_summary(stats: Dict) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)
    c_break = "#d62728"
    c_normal = "#2ca02c"
    c_mix = "#1f77b4"

    break_files = stats["break"]["files"]
    normal_files = stats["normal"]["files"]
    total_files = break_files + normal_files

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["파단", "정상", "전체"], [break_files, normal_files, total_files], color=[c_break, c_normal, c_mix])
    ax1.set_title("OUT processed CSV 수")

    def ratio(rows: List[int]) -> List[float]:
        total = sum(rows)
        return [(x / total * 100) if total else 0.0 for x in rows]

    ax2 = fig.add_subplot(gs[0, 1])
    rb = ratio(stats["break"]["degree_rows"])
    ax2.bar(DEGREE_LABELS, rb, color=c_break)
    ax2.set_title("파단 각도 비율(행 기준)")
    ax2.set_ylabel("%")

    ax3 = fig.add_subplot(gs[0, 2])
    rn = ratio(stats["normal"]["degree_rows"])
    ax3.bar(DEGREE_LABELS, rn, color=c_normal)
    ax3.set_title("정상 각도 비율(행 기준)")
    ax3.set_ylabel("%")

    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(DEGREE_LABELS))
    w = 0.35
    ax4.bar(x - w / 2, stats["break"]["degree_files"], w, color=c_break, label="파단")
    ax4.bar(x + w / 2, stats["normal"]["degree_files"], w, color=c_normal, label="정상")
    ax4.set_xticks(x)
    ax4.set_xticklabels(DEGREE_LABELS)
    ax4.set_title("각도 구간별 파일 수")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    hb = histogram_file_coverage(stats["break"]["file_heights"])
    ax5.bar(range(len(hb)), hb, color=c_break)
    ax5.set_xticks(range(len(HEIGHT_LABELS)))
    ax5.set_xticklabels(HEIGHT_LABELS, rotation=45, ha="right")
    ax5.set_title("파단 높이 커버리지(파일 수)")

    ax6 = fig.add_subplot(gs[1, 2])
    hn = histogram_file_coverage(stats["normal"]["file_heights"])
    ax6.bar(range(len(hn)), hn, color=c_normal)
    ax6.set_xticks(range(len(HEIGHT_LABELS)))
    ax6.set_xticklabels(HEIGHT_LABELS, rotation=45, ha="right")
    ax6.set_title("정상 높이 커버리지(파일 수)")

    ax7 = fig.add_subplot(gs[2, :])
    bh = histogram_break_heights(stats["break"]["break_heights"])
    ax7.bar(range(len(bh)), bh, color=c_break)
    ax7.set_xticks(range(len(HEIGHT_LABELS)))
    ax7.set_xticklabels(HEIGHT_LABELS, rotation=45, ha="right")
    ax7.set_title("파단 높이 분포(breakheight)")
    ax7.set_ylabel("건수")

    fig.suptitle("4. merge_data 통계", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="merge_data 통계 확인")
    parser.add_argument("--data-dir", default=str(MERGE_DATA_DIR), help="분석 대상 merge_data 경로")
    parser.add_argument("--no-plot", action="store_true", help="그래프 표시 생략")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = collect_stats(Path(args.data_dir))
    if stats is None:
        return
    print_summary(stats)
    if not args.no_plot:
        plot_summary(stats)


if __name__ == "__main__":
    main()
