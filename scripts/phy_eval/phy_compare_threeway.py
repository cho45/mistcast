#!/usr/bin/env python3
"""Compare baseline FSK / baseline DSSS / new DSSS metrics."""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
from typing import Dict, List, Optional, Tuple

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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline FSK / baseline DSSS / new DSSS from metrics CSV"
    )
    p.add_argument("--base", required=True, help="Baseline metrics csv (FSK + DSSS)")
    p.add_argument("--new", required=True, help="New metrics csv (DSSS required)")
    p.add_argument("--out", default="", help="Output CSV path")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    return p.parse_args()


def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def key_mode_scenario(row: Dict[str, str]) -> Tuple[str, str]:
    return row.get("mode", ""), row.get("scenario", "")


def key_mode_scenario_phy(row: Dict[str, str]) -> Tuple[str, str, str]:
    return row.get("mode", ""), row.get("scenario", ""), row.get("phy", "")


def to_float(v: Optional[str]) -> float:
    if v in ("", "NaN", "nan", "None", None):
        return float("nan")
    return float(v)


def to_int(v: Optional[str]) -> Optional[int]:
    if v in ("", "NaN", "nan", "None", None):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def fmt(v: float) -> str:
    if v != v:
        return "NaN"
    return f"{v:.6f}"


def extract_count(row: Optional[Dict[str, str]], count_key: str, prob_key: str) -> Tuple[Optional[int], Optional[int]]:
    if row is None:
        return None, None
    n = to_int(row.get("trials"))
    if n is None or n <= 0:
        return None, None
    x = to_int(row.get(count_key))
    if x is not None:
        return x, n
    p = to_float(row.get(prob_key))
    if p == p:
        return int(round(p * n)), n
    return None, None


def extract_ber_count(row: Optional[Dict[str, str]]) -> Tuple[Optional[int], Optional[int]]:
    if row is None:
        return None, None
    errs = to_int(row.get("total_bit_errors"))
    bits = to_int(row.get("total_bits_compared"))
    if errs is not None and bits is not None and bits > 0:
        return errs, bits
    ber = to_float(row.get("ber"))
    trials = to_int(row.get("trials"))
    payload_bits = None
    scenario = row.get("scenario", "")
    # scenario 文字列から bits= を拾う fallback
    marker = "bits="
    pos = scenario.find(marker)
    if pos >= 0:
        tail = scenario[pos + len(marker) :]
        num = ""
        for ch in tail:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            payload_bits = int(num)
    if ber == ber and trials and payload_bits:
        total = trials * payload_bits
        return int(round(ber * total)), total
    return None, None


def two_proportion_stats(
    x_base: Optional[int], n_base: Optional[int], x_new: Optional[int], n_new: Optional[int], alpha: float
) -> Tuple[float, float, float, float, str]:
    if (
        x_base is None
        or n_base is None
        or x_new is None
        or n_new is None
        or n_base <= 0
        or n_new <= 0
    ):
        return float("nan"), float("nan"), float("nan"), float("nan"), "insufficient"

    p_base = x_base / n_base
    p_new = x_new / n_new
    delta = p_new - p_base

    p_pool = (x_base + x_new) / (n_base + n_new)
    se_pool = (p_pool * (1.0 - p_pool) * (1.0 / n_base + 1.0 / n_new)) ** 0.5

    nd = statistics.NormalDist()
    if se_pool == 0.0:
        p_value = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / se_pool
        p_value = 2.0 * (1.0 - nd.cdf(abs(z)))

    se_unpooled = ((p_base * (1.0 - p_base) / n_base) + (p_new * (1.0 - p_new) / n_new)) ** 0.5
    z_crit = nd.inv_cdf(1.0 - alpha / 2.0)
    if se_unpooled == 0.0:
        ci_low = delta
        ci_high = delta
    else:
        ci_low = delta - z_crit * se_unpooled
        ci_high = delta + z_crit * se_unpooled

    significant = "yes" if (p_value < alpha and not (ci_low <= 0.0 <= ci_high)) else "no"
    return delta, p_value, ci_low, ci_high, significant


def main() -> int:
    args = parse_args()
    alpha = min(max(args.alpha, 1e-6), 0.5)

    base_rows = read_csv(pathlib.Path(args.base))
    new_rows = read_csv(pathlib.Path(args.new))

    base_map = {key_mode_scenario_phy(r): r for r in base_rows}
    new_map = {key_mode_scenario_phy(r): r for r in new_rows}

    scenario_keys = sorted(
        set((m, s) for (m, s, _) in base_map.keys()) | set((m, s) for (m, s, _) in new_map.keys())
    )

    out_rows: List[Dict[str, str]] = []
    for mode, scenario in scenario_keys:
        base_fsk = base_map.get((mode, scenario, "fsk"))
        base_dsss = base_map.get((mode, scenario, "dsss"))
        new_dsss = new_map.get((mode, scenario, "dsss"))

        status = "ok"
        if base_fsk is None:
            status = "missing_base_fsk"
        if base_dsss is None:
            status = "missing_base_dsss"
        if new_dsss is None:
            status = "missing_new_dsss"

        row: Dict[str, str] = {"mode": mode, "scenario": scenario, "status": status}

        for f in NUMERIC_FIELDS:
            bf = to_float((base_fsk or {}).get(f))
            bd = to_float((base_dsss or {}).get(f))
            nd = to_float((new_dsss or {}).get(f))
            row[f"baseline_fsk_{f}"] = fmt(bf)
            row[f"baseline_dsss_{f}"] = fmt(bd)
            row[f"new_dsss_{f}"] = fmt(nd)
            row[f"delta_new_vs_baseline_fsk_{f}"] = fmt(nd - bf if nd == nd and bf == bf else float("nan"))
            row[f"delta_new_vs_baseline_dsss_{f}"] = fmt(nd - bd if nd == nd and bd == bd else float("nan"))

        # p_complete_deadline significance
        x_bd, n_bd = extract_count(base_dsss, "deadline_hits", "p_complete_deadline")
        x_bf, n_bf = extract_count(base_fsk, "deadline_hits", "p_complete_deadline")
        x_nd, n_nd = extract_count(new_dsss, "deadline_hits", "p_complete_deadline")

        d_vs_bd = two_proportion_stats(x_bd, n_bd, x_nd, n_nd, alpha)
        d_vs_bf = two_proportion_stats(x_bf, n_bf, x_nd, n_nd, alpha)

        row["sig_p_complete_deadline_new_vs_baseline_dsss_delta"] = fmt(d_vs_bd[0])
        row["sig_p_complete_deadline_new_vs_baseline_dsss_p"] = fmt(d_vs_bd[1])
        row["sig_p_complete_deadline_new_vs_baseline_dsss_ci_low"] = fmt(d_vs_bd[2])
        row["sig_p_complete_deadline_new_vs_baseline_dsss_ci_high"] = fmt(d_vs_bd[3])
        row["sig_p_complete_deadline_new_vs_baseline_dsss"] = d_vs_bd[4]
        row["sig_p_complete_deadline_new_vs_baseline_fsk_delta"] = fmt(d_vs_bf[0])
        row["sig_p_complete_deadline_new_vs_baseline_fsk_p"] = fmt(d_vs_bf[1])
        row["sig_p_complete_deadline_new_vs_baseline_fsk_ci_low"] = fmt(d_vs_bf[2])
        row["sig_p_complete_deadline_new_vs_baseline_fsk_ci_high"] = fmt(d_vs_bf[3])
        row["sig_p_complete_deadline_new_vs_baseline_fsk"] = d_vs_bf[4]

        # BER significance (bit-level, if totals are available)
        e_bd, b_bd = extract_ber_count(base_dsss)
        e_bf, b_bf = extract_ber_count(base_fsk)
        e_nd, b_nd = extract_ber_count(new_dsss)
        ber_vs_bd = two_proportion_stats(e_bd, b_bd, e_nd, b_nd, alpha)
        ber_vs_bf = two_proportion_stats(e_bf, b_bf, e_nd, b_nd, alpha)

        row["sig_ber_new_vs_baseline_dsss_delta"] = fmt(ber_vs_bd[0])
        row["sig_ber_new_vs_baseline_dsss_p"] = fmt(ber_vs_bd[1])
        row["sig_ber_new_vs_baseline_dsss_ci_low"] = fmt(ber_vs_bd[2])
        row["sig_ber_new_vs_baseline_dsss_ci_high"] = fmt(ber_vs_bd[3])
        row["sig_ber_new_vs_baseline_dsss"] = ber_vs_bd[4]
        row["sig_ber_new_vs_baseline_fsk_delta"] = fmt(ber_vs_bf[0])
        row["sig_ber_new_vs_baseline_fsk_p"] = fmt(ber_vs_bf[1])
        row["sig_ber_new_vs_baseline_fsk_ci_low"] = fmt(ber_vs_bf[2])
        row["sig_ber_new_vs_baseline_fsk_ci_high"] = fmt(ber_vs_bf[3])
        row["sig_ber_new_vs_baseline_fsk"] = ber_vs_bf[4]

        out_rows.append(row)

    print(
        "mode,scenario,status,new_vs_base_dsss_delta_pcd,new_vs_base_dsss_p,new_vs_base_fsk_delta_pcd,new_vs_base_fsk_p"
    )
    for r in out_rows:
        print(
            ",".join(
                [
                    r["mode"],
                    r["scenario"],
                    r["status"],
                    r["sig_p_complete_deadline_new_vs_baseline_dsss_delta"],
                    r["sig_p_complete_deadline_new_vs_baseline_dsss_p"],
                    r["sig_p_complete_deadline_new_vs_baseline_fsk_delta"],
                    r["sig_p_complete_deadline_new_vs_baseline_fsk_p"],
                ]
            )
        )

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(out_rows[0].keys()) if out_rows else []
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if out_rows:
                w.writeheader()
                w.writerows(out_rows)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

