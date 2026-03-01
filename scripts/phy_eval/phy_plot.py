#!/usr/bin/env python3
"""Plot PHY metrics CSV with matplotlib."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import re
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot saved PHY metrics CSV")
    p.add_argument("--input", required=True, help="metrics csv path")
    p.add_argument("--out-dir", required=True, help="output directory for png")
    p.add_argument(
        "--metric",
        default="p_complete_deadline",
        help="metric for main curve (default: p_complete_deadline)",
    )
    p.add_argument(
        "--output",
        default="phy_summary.png",
        help="output png filename (default: phy_summary.png)",
    )
    return p.parse_args()


def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_param(scenario: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}=([-+0-9.]+)", scenario)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def to_float(v: str) -> float:
    if v in ("", "NaN", "nan", "None", None):
        return float("nan")
    return float(v)


def collect_line_points(
    rows: List[Dict[str, str]], mode_prefix: str, x_key: str, y_key: str
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    dsss = []
    fsk = []
    for r in rows:
        scenario = r.get("scenario", "")
        if not scenario.startswith(mode_prefix + "("):
            continue
        x = parse_param(scenario, x_key)
        y = to_float(r.get(y_key, "NaN"))
        if x is None or y != y:
            continue
        if r.get("phy") == "dsss":
            dsss.append((x, y))
        elif r.get("phy") == "fsk":
            fsk.append((x, y))

    dsss.sort(key=lambda t: t[0])
    fsk.sort(key=lambda t: t[0])
    return dsss, fsk


def plot_line_on_axis(ax, rows: List[Dict[str, str]], mode_prefix: str, x_key: str, y_key: str) -> None:
    dsss, fsk = collect_line_points(rows, mode_prefix, x_key, y_key)

    if not dsss and not fsk:
        ax.set_title(f"{mode_prefix}: {y_key} (no data)")
        ax.grid(True, alpha=0.2)
        return

    if dsss:
        ax.plot([x for x, _ in dsss], [y for _, y in dsss], marker="o", label="DSSS")
    if fsk:
        ax.plot([x for x, _ in fsk], [y for _, y in fsk], marker="o", label="FSK")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{mode_prefix}: {y_key}")
    ax.grid(True, alpha=0.3)
    ax.legend()


def collect_multipath_points(rows: List[Dict[str, str]], y_key: str) -> tuple[list[str], list[float], list[float]]:
    categories = ["none", "mild", "medium", "harsh"]
    dsss_vals = []
    fsk_vals = []

    for c in categories:
        d = float("nan")
        f = float("nan")
        for r in rows:
            scenario = r.get("scenario", "")
            if not scenario.startswith("multipath("):
                continue
            if f"profile={c}" not in scenario:
                continue
            y = to_float(r.get(y_key, "NaN"))
            if r.get("phy") == "dsss":
                d = y
            elif r.get("phy") == "fsk":
                f = y
        dsss_vals.append(d)
        fsk_vals.append(f)
    return categories, dsss_vals, fsk_vals


def plot_multipath_on_axis(ax, rows: List[Dict[str, str]], y_key: str) -> None:
    categories, dsss_vals, fsk_vals = collect_multipath_points(rows, y_key)

    if all(v != v for v in dsss_vals) and all(v != v for v in fsk_vals):
        ax.set_title(f"multipath: {y_key} (no data)")
        ax.grid(True, alpha=0.2)
        return

    x = range(len(categories))
    w = 0.38

    ax.bar([i - w / 2 for i in x], dsss_vals, width=w, label="DSSS")
    ax.bar([i + w / 2 for i in x], fsk_vals, width=w, label="FSK")
    ax.set_xticks(list(x), categories)
    ax.set_ylabel(y_key)
    ax.set_title(f"multipath: {y_key}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()


def main() -> int:
    args = parse_args()
    in_path = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sandbox/CI で matplotlib のキャッシュ先が書けないケースを回避
    cache_root = (in_path.resolve().parents[2] / "eval" / ".mplcache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    rows = read_csv(in_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib is required: {e}")
        return 2

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    fig.suptitle(f"PHY Summary: metric={args.metric}", fontsize=14)

    plot_line_on_axis(axes[0, 0], rows, "awgn", "sigma", args.metric)
    plot_line_on_axis(axes[1, 0], rows, "cfo", "cfo", args.metric)
    plot_line_on_axis(axes[2, 0], rows, "ppm", "ppm", args.metric)
    plot_line_on_axis(axes[3, 0], rows, "loss", "loss", args.metric)
    plot_multipath_on_axis(axes[4, 0], rows, args.metric)

    plot_line_on_axis(axes[0, 1], rows, "awgn", "sigma", "ber")
    plot_line_on_axis(axes[1, 1], rows, "cfo", "cfo", "ber")
    plot_line_on_axis(axes[2, 1], rows, "ppm", "ppm", "ber")
    plot_line_on_axis(axes[3, 1], rows, "loss", "loss", "ber")
    plot_multipath_on_axis(axes[4, 1], rows, "ber")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = out_dir / args.output
    fig.savefig(out_path)
    plt.close(fig)

    print(f"saved plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
