#!/usr/bin/env python3
"""Plot PHY metrics CSV with matplotlib."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import re
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot saved PHY metrics CSV")
    p.add_argument(
        "--input",
        required=True,
        action="append",
        help="metrics csv path (repeatable)",
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="legend prefix for each --input (repeatable, optional)",
    )
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
    p.add_argument(
        "--awgn-axis",
        default="snr-db",
        choices=["snr-db", "sigma", "sigma-db"],
        help="x-axis for awgn row (default: snr-db)",
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
) -> Dict[str, list[tuple[float, float]]]:
    series: Dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        scenario = r.get("scenario", "")
        if not scenario.startswith(mode_prefix + "("):
            continue
        if x_key == "snr_db":
            x = to_float(r.get("awgn_snr_db", "NaN"))
            if x != x:
                x = None
        elif x_key == "sigma_db":
            sigma = parse_param(scenario, "sigma")
            if sigma is None or sigma <= 0.0:
                x = None
            else:
                x = 20.0 * math.log10(sigma)
        else:
            x = parse_param(scenario, x_key)
        y = to_float(r.get(y_key, "NaN"))
        if x is None or y != y:
            continue
        phy = r.get("phy", "unknown")
        series.setdefault(phy, []).append((x, y))

    for points in series.values():
        points.sort(key=lambda t: t[0])
    return series


def build_series(
    datasets: Sequence[Tuple[str, List[Dict[str, str]]]],
    mode_prefix: str,
    x_key: str,
    y_key: str,
) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for label, rows in datasets:
        for phy, points in collect_line_points(rows, mode_prefix, x_key, y_key).items():
            out[f"{label}:{phy}"] = points
    return out


def plot_line_on_axis(
    ax,
    datasets: Sequence[Tuple[str, List[Dict[str, str]]]],
    mode_prefix: str,
    x_key: str,
    y_key: str,
) -> None:
    series = build_series(datasets, mode_prefix, x_key, y_key)

    if not series:
        ax.set_title(f"{mode_prefix}: {y_key} (no data)")
        ax.grid(True, alpha=0.2)
        return

    for name, points in sorted(series.items()):
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=name)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{mode_prefix}: {y_key}")
    ax.grid(True, alpha=0.3)
    ax.legend()


def collect_multipath_points(rows: List[Dict[str, str]], y_key: str) -> tuple[list[str], Dict[str, list[float]]]:
    categories = ["none", "mild", "medium", "harsh"]
    phys = sorted(
        {r.get("phy", "unknown") for r in rows if r.get("scenario", "").startswith("multipath(")}
    )
    values: Dict[str, list[float]] = {phy: [] for phy in phys}

    for c in categories:
        bucket = {phy: float("nan") for phy in phys}
        for r in rows:
            scenario = r.get("scenario", "")
            if not scenario.startswith("multipath("):
                continue
            if f"profile={c}" not in scenario:
                continue
            y = to_float(r.get(y_key, "NaN"))
            bucket[r.get("phy", "unknown")] = y
        for phy in phys:
            values[phy].append(bucket[phy])
    return categories, values


def plot_multipath_on_axis(
    ax,
    datasets: Sequence[Tuple[str, List[Dict[str, str]]]],
    y_key: str,
) -> None:
    categories = ["none", "mild", "medium", "harsh"]
    values: Dict[str, List[float]] = {}
    for label, rows in datasets:
        _, one = collect_multipath_points(rows, y_key)
        for phy, vals in one.items():
            values[f"{label}:{phy}"] = vals

    if not values or all(all(v != v for v in vals) for vals in values.values()):
        ax.set_title(f"multipath: {y_key} (no data)")
        ax.grid(True, alpha=0.2)
        return

    x = range(len(categories))
    n = len(values)
    w = 0.8 / max(n, 1)

    for idx, (phy, vals) in enumerate(sorted(values.items())):
        offset = (idx - (n - 1) / 2.0) * w
        ax.bar([i + offset for i in x], vals, width=w, label=phy)
    ax.set_xticks(list(x), categories)
    ax.set_ylabel(y_key)
    ax.set_title(f"multipath: {y_key}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()


def main() -> int:
    args = parse_args()
    in_paths = [pathlib.Path(p) for p in args.input]
    if args.label and len(args.label) != len(in_paths):
        print("label count must match input count")
        return 2
    if args.label:
        labels = list(args.label)
    else:
        labels = [p.stem for p in in_paths]
    # ラベル重複時は添字を付けてユニーク化
    seen: Dict[str, int] = {}
    for i, label in enumerate(labels):
        n = seen.get(label, 0)
        if n > 0:
            labels[i] = f"{label}#{n+1}"
        seen[label] = n + 1

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sandbox/CI で matplotlib のキャッシュ先が書けないケースを回避
    cache_root = pathlib.Path("dsp/eval/.mplcache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    datasets: List[Tuple[str, List[Dict[str, str]]]] = [
        (labels[i], read_csv(path)) for i, path in enumerate(in_paths)
    ]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib is required: {e}")
        return 2

    awgn_x = {
        "snr-db": "snr_db",
        "sigma-db": "sigma_db",
        "sigma": "sigma",
    }[args.awgn_axis]
    has_cfo = any(
        r.get("scenario", "").startswith("cfo(")
        for _, rows in datasets
        for r in rows
    )
    has_fading = any(
        r.get("scenario", "").startswith("fading(")
        for _, rows in datasets
        for r in rows
    )
    mode_rows: list[tuple[str, str | None]] = [("awgn", awgn_x)]
    if has_cfo:
        mode_rows.append(("cfo", "cfo"))
    mode_rows.append(("ppm", "ppm"))
    mode_rows.append(("loss", "loss"))
    if has_fading:
        mode_rows.append(("fading", "fade"))
    mode_rows.append(("multipath", None))

    fig, axes = plt.subplots(len(mode_rows), 2, figsize=(16, 4.2 * len(mode_rows)))
    fig.suptitle(f"PHY Summary: metric={args.metric}", fontsize=14)
    if len(mode_rows) == 1:
        axes = [axes]  # type: ignore[assignment]

    for i, (mode, x_key) in enumerate(mode_rows):
        ax_l = axes[i][0]  # type: ignore[index]
        ax_r = axes[i][1]  # type: ignore[index]
        if mode == "multipath":
            plot_multipath_on_axis(ax_l, datasets, args.metric)
            plot_multipath_on_axis(ax_r, datasets, "ber")
        else:
            assert x_key is not None
            plot_line_on_axis(ax_l, datasets, mode, x_key, args.metric)
            plot_line_on_axis(ax_r, datasets, mode, x_key, "ber")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = out_dir / args.output
    fig.savefig(out_path)
    plt.close(fig)

    print(f"saved plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
