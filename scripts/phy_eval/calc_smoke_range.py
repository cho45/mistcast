import math
import argparse
import json

# --- 制約条件の定義 ---
F_MIN = 50.0
F_MAX = 20000.0
ALPHA = 0.3
MAX_CFO = 10.0
MAX_DRIFT_PER_SYMBOL_DEG = 30.0
SF_CANDIDATES = [15, 31]
BITS_PER_SYMBOL_DATA = 2
BITS_PER_SYMBOL_SYNC = 1

def calc_total_symbols(payload_bytes, sf, sync_bits, preamble_repeat, ppb):
    # Packet Header: 6 bytes, CRC: 4 bytes -> total 10 bytes overhead
    # FEC: 2x rate
    raw_bits = (payload_bytes + 10) * 8 + 6
    fec_bits = raw_bits * 2
    data_symbols_per_packet = math.ceil(fec_bits / BITS_PER_SYMBOL_DATA)
    # Burst structure: Preamble + SyncWord + (DataSymbols * PPB)
    return preamble_repeat + sync_bits + (data_symbols_per_packet * ppb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "csv", "text"], default="text")
    args = parser.parse_args()

    max_bw = F_MAX - F_MIN
    max_rc = max_bw / (1.0 + ALPHA)
    
    rc_candidates = [2000, 4000, 6000, 8000, 10000, 12000]
    rc_candidates = [r for r in rc_candidates if r <= max_rc]
    
    # 同期構成のバリエーション
    # (sync_bits, preamble_repeat, packets_per_burst)
    sync_configs = [
        (32, 4, 2), # Legacy Default
        (16, 2, 2), # Optimized Sync, 2 Packets
        (16, 2, 1)  # Optimized Sync, 1 Packet (Highest Sync Frequency)
    ]
    
    candidates = []

    for sync_bits, pre_rep, ppb in sync_configs:
        for rc in rc_candidates:
            fc = 17500 # 基準
            for sf in SF_CANDIDATES:
                t_sym = sf / rc
                drift_per_sym = 360.0 * MAX_CFO * t_sym
                
                if drift_per_sym > MAX_DRIFT_PER_SYMBOL_DEG:
                    continue

                for p_size in [24]: # パケットサイズ固定
                    total_syms = calc_total_symbols(p_size, sf, sync_bits, pre_rep, ppb)
                    duration = total_syms * t_sym
                    theory_bps = (p_size * ppb * 8) / duration
                    
                    candidates.append({
                        "sync_bits": sync_bits,
                        "preamble_repeat": pre_rep,
                        "packets_per_burst": ppb,
                        "rc": rc,
                        "fc": int(fc),
                        "sf": sf,
                        "mseq_order": int(math.log2(sf + 1)),
                        "payload_bytes": p_size,
                        "duration_ms": int(duration * 1000),
                        "theory_bps": int(theory_bps),
                        "drift_per_sym": round(drift_per_sym, 1),
                        "total_drift": round(360.0 * MAX_CFO * duration, 1)
                    })

    candidates.sort(key=lambda x: x["theory_bps"], reverse=True)

    if args.format == "json":
        print(json.dumps(candidates))
    else:
        print(f"{'Sync':>4} | {'PPB':>3} | {'Rc':>5} | {'SF':>2} | {'Dur(ms)':>7} | {'BPS':>5} | {'D/Sym'}")
        print("-" * 60)
        for c in candidates:
            print(f"{c['sync_bits']:4d} | {c['packets_per_burst']:3d} | {c['rc']:5d} | {c['sf']:2d} | {c['duration_ms']:7d} | {c['theory_bps']:5d} | {c['drift_per_sym']:6.1f}")

if __name__ == "__main__":
    main()
