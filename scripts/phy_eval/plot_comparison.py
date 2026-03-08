#!/usr/bin/env python3
import subprocess
import json
import matplotlib.pyplot as plt
import sys
import os
import re

def run_eval(phy):
    """指定された PHY で評価を実行し、JSON 結果を返す"""
    cmd = [
        "cargo", "run", "--release", "--bin", "dsss_e2e_eval", "--",
        "--phy", phy,
        "--mode", "sweep-awgn",
        "--packets-per-frame", "3",
        "--sweep-awgn", "0.2,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.9",
        "--columns", "scenario,crc_pass_ratio,goodput_effective_bps,goodput_success_mean_bps,ebn0_db,raw_ber",
        "--output", "json"
    ]

    print(f"Running eval for {phy}...", file=sys.stderr)
    result = subprocess.run(cmd, cwd="dsp", capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running eval for {phy}: {result.stderr}", file=sys.stderr)
        return None

    # 各行の JSON をパース
    data_points = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            data = json.loads(line)
            data_points.append(data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON line: {e}", file=sys.stderr)
            continue

    return data_points

def main():
    # mary と dsss を実行
    results = {}
    for phy in ["mary", "dsss"]:
        results[phy] = run_eval(phy)
        if results[phy] is None:
            print(f"Failed to get results for {phy}", file=sys.stderr)
            return 1
        print(f"Got {len(results[phy])} data points for {phy}", file=sys.stderr)

    # グラフ描画（2x2サブプロット）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PHY Performance Comparison: MARY vs DSSS', fontsize=16)

    # 各サブプロットの設定
    metrics = [
        ('goodput_effective_bps', 'Goodput Effective (bps)', axes[0, 0]),
        ('goodput_success_mean_bps', 'Goodput Success Mean (bps)', axes[0, 1]),
        ('crc_pass_ratio', 'CRC Pass Ratio', axes[1, 0]),
        ('raw_ber', 'Raw BER', axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        # scenario でマッチングさせるため、まず sigma -> 各PHYのメトリクス のマッピングを作成
        sigma_to_mary = {}
        sigma_to_dsss = {}

        for point in results["mary"]:
            scenario = point.get('scenario', '')
            match = re.search(r'sigma=([\d.]+)', scenario)
            if match:
                sigma = float(match.group(1))
                ebn0_db = point.get('ebn0_db')
                if ebn0_db is not None:
                    sigma_to_mary[sigma] = (ebn0_db, point.get(metric, 0))

        for point in results["dsss"]:
            scenario = point.get('scenario', '')
            match = re.search(r'sigma=([\d.]+)', scenario)
            if match:
                sigma = float(match.group(1))
                ebn0_db = point.get('ebn0_db')
                if ebn0_db is not None:
                    sigma_to_dsss[sigma] = (ebn0_db, point.get(metric, 0))

        # 共通の sigma 値でデータをプロット
        common_sigmas = sorted(set(sigma_to_mary.keys()) & set(sigma_to_dsss.keys()))

        x_vals = []  # 共通のx軸
        mary_y_vals = []
        dsss_y_vals = []

        for sigma in common_sigmas:
            mary_ebn0, mary_val = sigma_to_mary[sigma]
            dsss_ebn0, dsss_val = sigma_to_dsss[sigma]

            # 同じscenarioなのでEb/N0は同じはず。MARYのを共通のx軸として使う
            x_vals.append(mary_ebn0)
            mary_y_vals.append(mary_val)
            dsss_y_vals.append(dsss_val)

        # ソート
        sorted_data = sorted(zip(x_vals, mary_y_vals, dsss_y_vals))
        x_sorted, mary_y_sorted, dsss_y_sorted = zip(*sorted_data)

        # プロット
        ax.plot(x_sorted, mary_y_sorted, marker='o', label='MARY', linewidth=2, markersize=6)
        ax.plot(x_sorted, dsss_y_sorted, marker='o', label='DSSS', linewidth=2, markersize=6)

        ax.set_xlabel('Eb/N0 (dB)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()

    # 出力先のディレクトリが存在することを確認
    os.makedirs('docs', exist_ok=True)
    output_path = 'docs/plot_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
