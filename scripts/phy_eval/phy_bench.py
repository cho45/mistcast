import subprocess
import csv
import json
import os
import sys
import math
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# --- 実用的な評価条件 ---
TRIALS = 20
SAMPLE_RATE = 48000
SIGMA_LIST = [0.1, 0.2, 0.3] # ノイズ耐性の段階
FIXED_CFO = 5.0
# 室内反射プロファイル (複数のタップでデッドゾーンを回避)
MULTIPATH_PROFILE = "0:1.0,100:0.6,250:0.4,480:0.3" 

def run_eval(params):
    rc = params["rc"]
    fc = params["fc"]
    m = params["mseq_order"]
    p_bytes = params["payload_bytes"]
    sigma = params["sigma"]
    sync_bits = params["sync_bits"]
    pre_rep = params["preamble_repeat"]
    
    # 評価ツールに同期ワード設定を渡すために DspConfig を調整
    # 現状の dsss_e2e_eval には --sync-word-bits がないので、
    # もし無ければ引数を追加するか、デフォルト値を想定して進める。
    # ここでは --sync-word-bits と --preamble-repeat があると仮定してコマンドを構築。
    cmd = [
        "cargo", "run", "--release", "--bin", "dsss_e2e_eval", "--",
        "--sample-rate", str(SAMPLE_RATE),
        "--trials", str(TRIALS),
        "--chip-rate", str(rc),
        "--carrier-freq", str(fc),
        "--mseq-order", str(m),
        "--payload-bytes", str(p_bytes),
        "--sigma", str(sigma),
        "--cfo-hz", str(FIXED_CFO),
        "--multipath", MULTIPATH_PROFILE,
        # 新しく追加する引数
        "--sync-word-bits", str(sync_bits),
        "--preamble-repeat", str(pre_rep),
        "--mode", "point"
    ]
    
    try:
        result = subprocess.run(cmd, cwd="dsp", capture_output=True, text=True, check=True)
        last_line = result.stdout.strip().split("\n")[-1]
        return f"{sync_bits},{pre_rep},{rc},{params['sf']},{p_bytes},{sigma},{params['theory_bps']},{last_line}"
    except subprocess.CalledProcessError:
        return None

def main():
    # calc_smoke_range.py から候補を取得
    calc_cmd = [sys.executable, "scripts/phy_eval/calc_smoke_range.py", "--format", "json"]
    res = subprocess.run(calc_cmd, capture_output=True, text=True, check=True)
    candidates = json.loads(res.stdout)
    
    tasks = []
    for c in candidates:
        for sigma in SIGMA_LIST:
            t = c.copy()
            t["sigma"] = sigma
            tasks.append(t)
    
    print(f"Benchmarking {len(tasks)} combinations...")
    raw_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, res in enumerate(executor.map(run_eval, tasks)):
            if res: raw_results.append(res)
            if (i+1) % 10 == 0: print(f"Progress: {i+1}/{len(tasks)}")

    output_base = "scripts/phy_eval/bench_results"
    
    data = []
    with open(output_base + ".csv", "w", newline="") as f:
        f.write("sync_bits,pre_rep,rc,sf,p_bytes,sigma,theory_bps,raw_line\n")
        for line in raw_results:
            f.write(line + "\n")
            cols = line.split(",")
            try:
                data.append({
                    "sync_bits": int(cols[0]),
                    "sf": int(cols[3]),
                    "rc": int(cols[2]),
                    "sigma": float(cols[5]),
                    "theory_bps": float(cols[6]),
                    "p_complete": float(cols[-14]) # p_complete: 末尾から14番目
                })
            except: continue

    # グラフ生成
    print("Generating Comparative Analysis Plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for i, sigma in enumerate(SIGMA_LIST):
        ax = axes[i]
        subset = [d for d in data if d["sigma"] == sigma]
        
        # 比較軸：(SF, SyncBits) の組み合わせごとにラインを引く
        configs = [(15, 32), (15, 16), (31, 32), (31, 16)]
        for sf, sbits in configs:
            plot_data = sorted([d for d in subset if d["sf"] == sf and d["sync_bits"] == sbits], key=lambda x: x["theory_bps"])
            if not plot_data: continue
            
            bps = [d["theory_bps"] for d in plot_data]
            p_comp = [d["p_complete"] for d in plot_data]
            
            label = f"SF={sf}, Sync={sbits}b"
            ax.plot(bps, p_comp, marker='o', label=label, alpha=0.8)
            
            # 各点に Rc を注釈
            for d in plot_data:
                ax.annotate(f"{int(d['rc'])}", (d["theory_bps"], d["p_complete"]), 
                            xytext=(0, 5), textcoords="offset points", ha='center', fontsize=7)

        ax.set_title(f"Noise Sigma = {sigma}")
        ax.set_xlabel("Effective Throughput (bps)")
        if i == 0: ax.set_ylabel("Reliability (P-Complete)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.2)
        ax.legend(fontsize=8)

    plt.suptitle(f"DSSS Optimization: Impact of Sync Length and SF on Performance\n(CFO=5Hz, Complex Multipath)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_base + ".png")
    print(f"Benchmark finished. Comparative plot saved to {output_base}.png")

if __name__ == "__main__":
    main()
