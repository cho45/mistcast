#!/usr/bin/env python3
import argparse
import bisect
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


def sidecar_path_for(profile_path: Path) -> Path | None:
    name = profile_path.name
    candidates: list[Path] = []
    if name.endswith(".json.gz"):
        base = name[: -len(".json.gz")]
        candidates.append(profile_path.with_name(f"{base}.syms.json"))
    if name.endswith(".json"):
        base = name[: -len(".json")]
        candidates.append(profile_path.with_name(f"{base}.syms.json"))
    candidates.append(profile_path.with_name(f"{name}.syms.json"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


class SymsResolver:
    def __init__(
        self,
        range_tables: dict[str, tuple[list[int], list[tuple[int, int, str]]]],
        exact_tables: dict[str, dict[int, str]],
    ) -> None:
        self.range_tables = range_tables
        self.exact_tables = exact_tables

    @classmethod
    def from_sidecar(cls, syms_path: Path) -> "SymsResolver | None":
        try:
            with syms_path.open("r", encoding="utf-8") as f:
                sidecar = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        string_table = sidecar.get("string_table")
        data = sidecar.get("data")
        if not isinstance(string_table, list) or not isinstance(data, list):
            return None

        range_tables: dict[str, tuple[list[int], list[tuple[int, int, str]]]] = {}
        exact_tables: dict[str, dict[int, str]] = {}

        for image in data:
            if not isinstance(image, dict):
                continue
            lib_name = image.get("debug_name")
            symtab = image.get("symbol_table")
            known = image.get("known_addresses")
            if not isinstance(lib_name, str) or not isinstance(symtab, list):
                continue

            ranges: list[tuple[int, int, str]] = []
            symbols_by_idx: list[str | None] = []
            for sym in symtab:
                if not isinstance(sym, dict):
                    symbols_by_idx.append(None)
                    continue
                rva = sym.get("rva")
                size = sym.get("size")
                sym_idx = sym.get("symbol")
                if (
                    not isinstance(rva, int)
                    or not isinstance(size, int)
                    or not isinstance(sym_idx, int)
                    or sym_idx < 0
                    or sym_idx >= len(string_table)
                ):
                    symbols_by_idx.append(None)
                    continue
                sym_name = string_table[sym_idx]
                if not isinstance(sym_name, str):
                    symbols_by_idx.append(None)
                    continue
                end = rva + size if size > 0 else rva + 1
                ranges.append((rva, end, sym_name))
                symbols_by_idx.append(sym_name)

            if ranges:
                ranges.sort(key=lambda x: x[0])
                starts = [r[0] for r in ranges]
                range_tables[lib_name] = (starts, ranges)

            if isinstance(known, list) and symbols_by_idx:
                exact: dict[int, str] = {}
                for row in known:
                    if (
                        isinstance(row, list)
                        and len(row) >= 2
                        and isinstance(row[0], int)
                        and isinstance(row[1], int)
                    ):
                        addr = row[0]
                        sym_row_idx = row[1]
                        if 0 <= sym_row_idx < len(symbols_by_idx):
                            sym_name = symbols_by_idx[sym_row_idx]
                            if isinstance(sym_name, str):
                                exact[addr] = sym_name
                if exact:
                    exact_tables[lib_name] = exact

        if not range_tables and not exact_tables:
            return None
        return cls(range_tables, exact_tables)

    def resolve(self, lib_name: str, rel_addr: int) -> str | None:
        exact = self.exact_tables.get(lib_name)
        if exact is not None:
            sym = exact.get(rel_addr)
            if sym:
                return sym

        table = self.range_tables.get(lib_name)
        if table is None:
            return None
        starts, ranges = table
        idx = bisect.bisect_right(starts, rel_addr) - 1
        if idx < 0:
            return None
        start, end, name = ranges[idx]
        if start <= rel_addr < end:
            return name
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    with args.profile.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    syms_path = sidecar_path_for(args.profile)
    syms_resolver = SymsResolver.from_sidecar(syms_path) if syms_path else None

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
    if syms_path:
        state = "loaded" if syms_resolver is not None else "invalid"
        print(f"[profile-summary] syms={syms_path} ({state})")
    else:
        print("[profile-summary] syms=<none>")

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
        if rel_addr >= 0:
            if syms_resolver is not None:
                sym = syms_resolver.resolve(lib_name, rel_addr)
                if sym:
                    label = sym
            if (
                label == f"{lib_name} + 0x{rel_addr:x}"
                and lib_name == "dsss_e2e_eval"
                and binary_path
            ):
                sym = symbolize_macho(binary_path, rel_addr)
                if sym:
                    label = sym
        print(f"  {pct:6.2f}%  {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
