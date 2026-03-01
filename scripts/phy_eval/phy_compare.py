#!/usr/bin/env python3
"""Compare two saved PHY metrics CSV files."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List, Tuple

NUMERIC_FIELDS = [
    "p_complete",
    "p_complete_deadline",
    "ber",
    "per",
    "fer",
    "goodput_effective_bps",
    "goodput_success_mean_bps",
    "p95_complete_s",
    "mean_complete_s",
    "tx_signal_power",
    "awgn_noise_power",
    "awgn_snr_db",
]

KEY_FIELDS = ["mode", "scenario", "phy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two PHY benchmark CSV files")
    p.add_argument("--base", required=True, help="Base metrics csv")
    p.add_argument("--new", required=True, help="New metrics csv")
    p.add_argument("--out", default="", help="Output CSV path (optional)")
    return p.parse_args()


def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def to_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return tuple(row.get(k, "") for k in KEY_FIELDS)  # type: ignore[return-value]


def to_float(v: str) -> float:
    if v in ("", "NaN", "nan", "None", None):
        return float("nan")
    return float(v)


def fmt(v: float) -> str:
    if v != v:
        return "NaN"
    return f"{v:.6f}"


def main() -> int:
    args = parse_args()
    base_path = pathlib.Path(args.base)
    new_path = pathlib.Path(args.new)

    base_rows = read_csv(base_path)
    new_rows = read_csv(new_path)

    base_map = {to_key(r): r for r in base_rows}
    new_map = {to_key(r): r for r in new_rows}

    keys = sorted(set(base_map.keys()) | set(new_map.keys()))
    out_rows: List[Dict[str, str]] = []

    for k in keys:
        b = base_map.get(k)
        n = new_map.get(k)
        row: Dict[str, str] = {
            "mode": k[0],
            "scenario": k[1],
            "phy": k[2],
            "status": "both" if (b and n) else ("new_only" if n else "base_only"),
        }

        for f in NUMERIC_FIELDS:
            bv = to_float((b or {}).get(f, "NaN"))
            nv = to_float((n or {}).get(f, "NaN"))
            dv = nv - bv if (bv == bv and nv == nv) else float("nan")
            row[f"base_{f}"] = fmt(bv)
            row[f"new_{f}"] = fmt(nv)
            row[f"delta_{f}"] = fmt(dv)

        out_rows.append(row)

    # Console summary
    print("mode,scenario,phy,delta_p_complete_deadline,delta_ber,delta_goodput_effective_bps")
    for r in out_rows:
        if r["status"] != "both":
            continue
        print(
            ",".join(
                [
                    r["mode"],
                    r["scenario"],
                    r["phy"],
                    r["delta_p_complete_deadline"],
                    r["delta_ber"],
                    r["delta_goodput_effective_bps"],
                ]
            )
        )

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
            if out_rows:
                w.writeheader()
                w.writerows(out_rows)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
