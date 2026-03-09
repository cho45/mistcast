#!/usr/bin/env python3
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def safe_get(arr, idx, default=None):
    if idx is None:
        return default
    if idx < 0 or idx >= len(arr):
        return default
    return arr[idx]


def symbolize_macho(binary_path: Path, rel_addr: int) -> str | None:
    # Samply JSON keeps Mach-O addresses as image-relative offsets.
    # Typical executable base for Mach-O PIE on macOS is 0x100000000.
    abs_addr = 0x100000000 + rel_addr
    cmd = [
        "xcrun",
        "atos",
        "-o",
        str(binary_path),
        "-l",
        "0x100000000",
        hex(abs_addr),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    out = p.stdout.strip()
    if not out:
        return None
    return out.splitlines()[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    with args.profile.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    thread = profile["threads"][0]
    libs = profile.get("libs", [])
    sample_stacks = thread["samples"]["stack"]
    sample_weights = thread["samples"].get("weight", [])
    stack_frame = thread["stackTable"]["frame"]
    frame_func = thread["frameTable"]["func"]
    frame_addr = thread["frameTable"]["address"]
    func_res = thread["funcTable"]["resource"]
    res_lib = thread["resourceTable"]["lib"]

    self_by_frame = defaultdict(float)
    self_by_lib = defaultdict(float)
    total_weight = 0.0

    for i, stack_idx in enumerate(sample_stacks):
        if stack_idx is None or stack_idx < 0:
            continue
        w = 1.0
        if i < len(sample_weights):
            ww = sample_weights[i]
            if ww is not None:
                w = float(ww)
        total_weight += w

        frame_idx = safe_get(stack_frame, stack_idx, -1)
        if frame_idx is None or frame_idx < 0:
            continue
        func_idx = safe_get(frame_func, frame_idx, -1)
        if func_idx is None or func_idx < 0:
            continue
        res_idx = safe_get(func_res, func_idx, -1)
        lib_idx = safe_get(res_lib, res_idx, -1) if res_idx is not None and res_idx >= 0 else -1
        lib_name = "<unknown>"
        if lib_idx is not None and lib_idx >= 0:
            lib = safe_get(libs, lib_idx, {})
            lib_name = lib.get("name", "<unknown>")

        rel_addr = safe_get(frame_addr, frame_idx, -1)
        if rel_addr is None:
            rel_addr = -1
        key = (lib_name, int(rel_addr))
        self_by_frame[key] += w
        self_by_lib[lib_name] += w

    if total_weight <= 0:
        print("[profile-summary] no samples")
        return 1

    print(f"[profile-summary] file={args.profile}")
    print(f"[profile-summary] samples={int(total_weight)}")

    top_libs = sorted(self_by_lib.items(), key=lambda x: x[1], reverse=True)[:8]
    print("[profile-summary] top self libs:")
    for lib_name, w in top_libs:
        print(f"  {100.0 * w / total_weight:6.2f}%  {lib_name}")

    binary_path = None
    for lib in libs:
        if lib.get("name") == "dsss_e2e_eval":
            binary_path = Path(lib.get("path", ""))
            break

    top_frames = sorted(self_by_frame.items(), key=lambda x: x[1], reverse=True)[: args.top]
    print("[profile-summary] top self frames:")
    for (lib_name, rel_addr), w in top_frames:
        pct = 100.0 * w / total_weight
        label = f"{lib_name} + 0x{rel_addr:x}"
        if lib_name == "dsss_e2e_eval" and binary_path and rel_addr >= 0:
            sym = symbolize_macho(binary_path, rel_addr)
            if sym:
                label = sym
        print(f"  {pct:6.2f}%  {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
