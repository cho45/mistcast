#!/usr/bin/env python3
"""Run DSSS Encoder->Decoder e2e evaluation and persist metrics/metadata."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
DSP_MANIFEST = ROOT / "dsp" / "Cargo.toml"
DEFAULT_MODES = ["sweep-awgn", "sweep-ppm", "sweep-loss", "sweep-fading", "sweep-multipath"]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run and save DSSS e2e evaluation metrics")
    p.add_argument("--name", default=None, help="Run name (default: timestamp + commit)")
    p.add_argument("--out-dir", default=str(ROOT / "dsp" / "eval" / "runs"), help="Output directory")
    p.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated eval modes")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--payload-bytes", type=int, default=64)
    p.add_argument("--deadline-sec", type=float, default=0.8)
    p.add_argument("--max-sec", type=float, default=2.0)
    p.add_argument("--chunk-samples", type=int, default=16384)
    p.add_argument("--gap-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0xA11CE)
    p.add_argument("--target-p", type=float, default=0.95)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--cfo-hz", type=float, default=0.0)
    p.add_argument("--ppm", type=float, default=0.0)
    p.add_argument("--burst-loss", type=float, default=0.0)
    p.add_argument("--fading-depth", type=float, default=0.0)
    p.add_argument("--multipath", default="none")
    p.add_argument("--sweep-awgn", default="")
    p.add_argument("--sweep-cfo", default="")
    p.add_argument("--sweep-ppm", default="")
    p.add_argument("--sweep-loss", default="")
    p.add_argument("--sweep-fading", default="")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level for trial recommendation")
    p.add_argument("--power", type=float, default=0.8, help="Target power for trial recommendation")
    p.add_argument(
        "--min-effect",
        type=float,
        default=0.20,
        help="Minimum detectable absolute effect size for proportion metrics",
    )
    p.add_argument("--min-trials", type=int, default=30, help="Hard lower bound of trials")
    p.add_argument(
        "--auto-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically raise trials to statistically meaningful minimum",
    )
    p.add_argument("--release", action="store_true", default=True)
    return p.parse_args()


def recommended_trials(alpha: float, power: float, min_effect: float) -> int:
    alpha = min(max(alpha, 1e-6), 0.5)
    power = min(max(power, 0.5), 0.999999)
    min_effect = min(max(min_effect, 1e-3), 0.999)

    p1 = max(0.0, min(1.0, 0.5 - min_effect / 2.0))
    p2 = max(0.0, min(1.0, 0.5 + min_effect / 2.0))
    p_bar = 0.5 * (p1 + p2)

    nd = statistics.NormalDist()
    z_alpha = nd.inv_cdf(1.0 - alpha / 2.0)
    z_beta = nd.inv_cdf(power)

    term1 = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    term2 = z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    n = math.ceil(((term1 + term2) ** 2) / (min_effect**2))
    return max(n, 1)


def build_eval_cmd(mode: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        "cargo",
        "run",
        "--manifest-path",
        str(DSP_MANIFEST),
    ]
    if args.release:
        cmd.append("--release")
    cmd += [
        "--bin",
        "dsss_e2e_eval",
        "--",
        "--mode",
        mode,
        "--trials",
        str(args.trials),
        "--payload-bytes",
        str(args.payload_bytes),
        "--deadline-sec",
        str(args.deadline_sec),
        "--max-sec",
        str(args.max_sec),
        "--chunk-samples",
        str(args.chunk_samples),
        "--gap-samples",
        str(args.gap_samples),
        "--seed",
        str(args.seed),
        "--target-p",
        str(args.target_p),
        "--sigma",
        str(args.sigma),
        "--cfo-hz",
        str(args.cfo_hz),
        "--ppm",
        str(args.ppm),
        "--burst-loss",
        str(args.burst_loss),
        "--fading-depth",
        str(args.fading_depth),
        "--multipath",
        args.multipath,
    ]
    if args.sweep_awgn:
        cmd += ["--sweep-awgn", args.sweep_awgn]
    if args.sweep_cfo:
        cmd += ["--sweep-cfo", args.sweep_cfo]
    if args.sweep_ppm:
        cmd += ["--sweep-ppm", args.sweep_ppm]
    if args.sweep_loss:
        cmd += ["--sweep-loss", args.sweep_loss]
    if args.sweep_fading:
        cmd += ["--sweep-fading", args.sweep_fading]
    return cmd


def run_mode(mode: str, args: argparse.Namespace) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
    cmd = build_eval_cmd(mode, args)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    out_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        print(line, end="", flush=True)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    ret = proc.wait()
    stdout = "".join(out_lines)
    if ret != 0:
        raise RuntimeError(f"dsss_e2e_eval failed for mode={mode}\nstdout:\n{stdout}\nstderr:\n{stderr}")

    rows: List[Dict[str, str]] = []
    limits: List[Dict[str, str]] = []
    header: List[str] | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("scenario,"):
            header = next(csv.reader([line]))
            continue
        if line.startswith("# awgn_limit"):
            m = re.search(
                r"target_p_complete_deadline>=(?P<target>[0-9.]+), deadline_s=(?P<deadline>[0-9.]+)\) dsss=(?P<dsss>[^ ]+)",
                line,
            )
            if m:
                limits.append(
                    {
                        "mode": mode,
                        "target_p_complete_deadline": m.group("target"),
                        "deadline_s": m.group("deadline"),
                        "dsss_awgn_limit": m.group("dsss"),
                    }
                )
            continue
        if header and "," in line and not line.startswith("Running `"):
            parts = [p.strip() for p in line.split(",")]
            fixed_tail = len(header) - 1
            if len(parts) >= len(header):
                scenario = ",".join(parts[: len(parts) - fixed_tail])
                tail = parts[len(parts) - fixed_tail :]
                vals = [scenario] + tail
                rows.append(dict(zip(header, vals)))

    return stdout, rows, limits


def write_csv(path: pathlib.Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    requested_trials = args.trials

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_trials = recommended_trials(args.alpha, args.power, args.min_effect)
    if args.auto_trials:
        args.trials = max(args.trials, args.min_trials, rec_trials)
    else:
        args.trials = max(args.trials, args.min_trials)

    commit = git_commit()
    ts = now_utc()
    name = args.name or f"{ts}-{commit[:8]}"

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("no modes", file=sys.stderr)
        return 2

    all_rows: List[Dict[str, str]] = []
    all_limits: List[Dict[str, str]] = []
    raw_logs: Dict[str, str] = {}

    for mode in modes:
        print(f"[dsss_e2e_bench] running mode={mode}", flush=True)
        raw, rows, limits = run_mode(mode, args)
        raw_logs[mode] = raw
        for r in rows:
            r["run_id"] = name
            r["mode"] = mode
            r["git_commit"] = commit
            r["created_at_utc"] = ts
            all_rows.append(r)
        for l in limits:
            l["run_id"] = name
            l["git_commit"] = commit
            l["created_at_utc"] = ts
            all_limits.append(l)

    metrics_path = out_dir / f"{name}_metrics.csv"
    limits_path = out_dir / f"{name}_limits.csv"
    meta_path = out_dir / f"{name}_meta.json"
    log_path = out_dir / f"{name}_raw.log"

    write_csv(metrics_path, all_rows)
    write_csv(limits_path, all_limits)

    meta = {
        "run_id": name,
        "git_commit": commit,
        "created_at_utc": ts,
        "modes": modes,
        "requested_trials": requested_trials,
        "effective_trials": args.trials,
        "trials_policy": {
            "auto_trials": args.auto_trials,
            "alpha": args.alpha,
            "power": args.power,
            "min_effect": args.min_effect,
            "min_trials": args.min_trials,
            "recommended_trials": rec_trials,
        },
        "command_args": vars(args),
        "metrics_csv": str(metrics_path),
        "limits_csv": str(limits_path),
        "raw_log": str(log_path),
        "cwd": str(ROOT),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as f:
        for mode in modes:
            f.write(f"===== mode={mode} =====\n")
            f.write(raw_logs.get(mode, ""))
            f.write("\n")

    latest_link = out_dir / "latest_metrics.csv"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(metrics_path.name)
    except OSError:
        latest_link.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[dsss_e2e_bench] saved metrics: {metrics_path}", flush=True)
    print(f"[dsss_e2e_bench] saved limits : {limits_path}", flush=True)
    print(f"[dsss_e2e_bench] saved meta   : {meta_path}", flush=True)
    print(f"[dsss_e2e_bench] saved log    : {log_path}", flush=True)
    print(
        "[dsss_e2e_bench] trials policy: "
        f"requested={requested_trials} "
        f"effective={args.trials} "
        f"recommended={rec_trials} "
        f"(alpha={args.alpha}, power={args.power}, min_effect={args.min_effect})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
