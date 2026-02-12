#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""processed CSV(height/degree/x/y/z)를 2D 등고선 이미지로 저장한다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    for font_name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            plt.rcParams["font.family"] = font_name
            return
        except Exception:
            continue


def _calc_vrange(values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return None, None
    data_min = float(np.min(valid))
    data_max = float(np.max(valid))
    if data_max <= data_min:
        return data_min, data_max

    median = float(np.median(valid))
    std = float(np.std(valid))
    if std <= 0:
        return data_min, data_max

    p2 = float(np.percentile(valid, 2))
    p98 = float(np.percentile(valid, 98))
    vmin = max(data_min, median - 2.5 * std)
    vmax = min(data_max, median + 2.5 * std)
    if vmax - vmin < 1e-6:
        vmin, vmax = p2, p98
    if vmax - vmin < 1e-6:
        vmin, vmax = data_min, data_max
    return vmin, vmax


def plot_csv_2d(csv_file: str, output_file: Optional[str] = None, break_info_file: Optional[str] = None) -> Optional[Path]:
    """
    CSV를 읽어 X/Y/Z 3개 subplot 등고선 이미지를 만든다.
    `break_info_file` 인자는 기존 호출부 호환을 위해 유지한다.
    """
    _ = break_info_file
    configure_font()

    csv_path = Path(csv_file)
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None

    required = {"height", "degree", "x_value", "y_value", "z_value"}
    if not required.issubset(df.columns):
        return None

    heights = sorted(df["height"].unique())
    degrees = sorted(df["degree"].unique())
    mesh_d, mesh_h = np.meshgrid(degrees, heights)

    grids = []
    for col in ("x_value", "y_value", "z_value"):
        grid = (
            df.pivot_table(index="height", columns="degree", values=col, aggfunc="mean")
            .reindex(index=heights, columns=degrees)
            .to_numpy(dtype=np.float32)
        )
        grids.append(grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, grid, title in zip(axes, grids, ("X Value", "Y Value", "Z Value")):
        vmin, vmax = _calc_vrange(grid)
        levels = 30
        ax.contourf(
            mesh_d,
            mesh_h,
            grid,
            levels=levels,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            antialiased=True,
            extend="both",
        )
        ax.contour(mesh_d, mesh_h, grid, levels=levels, colors="black", linewidths=0.5, alpha=0.3)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Height (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    out_path = Path(output_file) if output_file else csv_path.with_name(csv_path.stem + "_2d_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="processed CSV를 2D 등고선 이미지로 변환")
    parser.add_argument("csv_file", help="입력 CSV 경로")
    parser.add_argument("--output-file", default=None, help="출력 PNG 경로")
    parser.add_argument("--break-info-file", default=None, help="호환용 인자(사용하지 않음)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = plot_csv_2d(args.csv_file, args.output_file, args.break_info_file)
    if out is None:
        raise SystemExit(1)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
