#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""raw_pole_data 수집 결과를 서버 단위로 요약/시각화한다."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CURRENT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = CURRENT_DIR / "3. raw_pole_data"
PROJECT_INFO_DIR = CURRENT_DIR / "1. project_info_list"
ANAL_POLE_DIR = CURRENT_DIR / "2. anal_pole_list"

SERVERS = {
    "main": "메인서버",
    "is": "이수서버",
    "kh": "건화서버",
}


def latest_file(pattern: str) -> Optional[Path]:
    files = glob.glob(pattern)
    if not files:
        return None
    return Path(max(files, key=lambda p: Path(p).stat().st_mtime))


def load_server_stats() -> Optional[Dict]:
    """JSON 기준 통계를 로드하고 raw_pole_data 파일 개수를 합친다."""
    anal_json = latest_file(str(ANAL_POLE_DIR / "anal2_poles_all_*.json"))
    if anal_json is None:
        print(f"오류: anal2_poles_all_*.json 파일이 없습니다. ({ANAL_POLE_DIR})")
        return None

    with anal_json.open("r", encoding="utf-8") as f:
        anal_payload = json.load(f)

    server_stats: Dict[str, Dict] = {
        s: {
            "projects_count": 0,
            "poles_break": 0,
            "poles_normal": 0,
            "csv_files": 0,
            "out_files": 0,
            "in_files": 0,
        }
        for s in SERVERS
    }

    project_to_server: Dict[str, str] = {}
    for server, payload in anal_payload.get("servers", {}).items():
        projects = payload.get("projects", {})
        if not isinstance(projects, dict):
            continue
        server_stats[server]["projects_count"] = len(projects)
        for project_name, info in projects.items():
            project_to_server[project_name] = server
            server_stats[server]["poles_break"] += int(info.get("break_count", 0))
            server_stats[server]["poles_normal"] += int(info.get("normal_count", 0))

    if RAW_DATA_DIR.exists():
        for category in ("break", "normal"):
            cat_dir = RAW_DATA_DIR / category
            if not cat_dir.exists():
                continue
            for project_dir in cat_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                server = project_to_server.get(project_dir.name)
                if server not in server_stats:
                    continue
                for pole_dir in project_dir.iterdir():
                    if not pole_dir.is_dir():
                        continue
                    for csv_path in pole_dir.glob("*.csv"):
                        name = csv_path.name.upper()
                        server_stats[server]["csv_files"] += 1
                        if "_OUT_" in name:
                            server_stats[server]["out_files"] += 1
                        elif "_IN_" in name:
                            server_stats[server]["in_files"] += 1

    proj_json = latest_file(str(PROJECT_INFO_DIR / "project_list_all_*.json"))
    if proj_json is not None:
        print(f"프로젝트 목록 JSON: {proj_json}")
    print(f"분석 전주 JSON: {anal_json}")
    return server_stats


def print_stats(stats: Dict) -> None:
    print("\n" + "=" * 100)
    print("raw_pole_data 서버별 요약")
    print("=" * 100)
    print(
        f"{'서버':<12} {'프로젝트':>8} {'전주':>10} {'CSV':>10} {'OUT':>10} {'IN':>10} {'파단전주':>10} {'정상전주':>10}"
    )
    print("-" * 100)

    total = {
        "projects_count": 0,
        "poles": 0,
        "csv_files": 0,
        "out_files": 0,
        "in_files": 0,
        "poles_break": 0,
        "poles_normal": 0,
    }

    for server in SERVERS:
        s = stats.get(server, {})
        poles = int(s.get("poles_break", 0)) + int(s.get("poles_normal", 0))
        print(
            f"{SERVERS[server]:<12} "
            f"{int(s.get('projects_count', 0)):>8} {poles:>10,} {int(s.get('csv_files', 0)):>10,} "
            f"{int(s.get('out_files', 0)):>10,} {int(s.get('in_files', 0)):>10,} "
            f"{int(s.get('poles_break', 0)):>10,} {int(s.get('poles_normal', 0)):>10,}"
        )

        total["projects_count"] += int(s.get("projects_count", 0))
        total["poles"] += poles
        total["csv_files"] += int(s.get("csv_files", 0))
        total["out_files"] += int(s.get("out_files", 0))
        total["in_files"] += int(s.get("in_files", 0))
        total["poles_break"] += int(s.get("poles_break", 0))
        total["poles_normal"] += int(s.get("poles_normal", 0))

    print("-" * 100)
    print(
        f"{'전체':<12} "
        f"{total['projects_count']:>8} {total['poles']:>10,} {total['csv_files']:>10,} "
        f"{total['out_files']:>10,} {total['in_files']:>10,} "
        f"{total['poles_break']:>10,} {total['poles_normal']:>10,}"
    )
    print("=" * 100)


def plot_stats(stats: Dict) -> None:
    servers = list(SERVERS.keys())
    labels = [SERVERS[s] for s in servers]
    projects = [int(stats[s]["projects_count"]) for s in servers]
    poles = [int(stats[s]["poles_break"]) + int(stats[s]["poles_normal"]) for s in servers]
    breaks = [int(stats[s]["poles_break"]) for s in servers]
    normals = [int(stats[s]["poles_normal"]) for s in servers]
    csvs = [int(stats[s]["csv_files"]) for s in servers]
    outs = [int(stats[s]["out_files"]) for s in servers]
    ins = [int(stats[s]["in_files"]) for s in servers]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(labels, projects, color=colors)
    ax1.set_title("프로젝트 수")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(labels, poles, color=colors)
    ax2.set_title("분석 완료 전주 수")

    ax3 = fig.add_subplot(gs[0, 2])
    x = range(len(labels))
    ax3.bar(x, breaks, label="파단", color="#d62728")
    ax3.bar(x, normals, bottom=breaks, label="정상", color="#2ca02c")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels)
    ax3.set_title("파단/정상 전주")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(labels, csvs, color=colors)
    ax4.set_title("수집 CSV 수")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(labels, outs, color=colors)
    ax5.set_title("OUT CSV 수")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(labels, ins, color=colors)
    ax6.set_title("IN CSV 수")

    fig.suptitle("3. raw_pole_data 서버별 통계", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw_pole_data 통계 확인")
    parser.add_argument("--no-plot", action="store_true", help="그래프를 띄우지 않고 표만 출력")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load_server_stats()
    if stats is None:
        return
    print_stats(stats)
    if not args.no_plot:
        plot_stats(stats)


if __name__ == "__main__":
    main()
