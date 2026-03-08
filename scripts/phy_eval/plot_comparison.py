#!/usr/bin/env python3
import subprocess
import json
import matplotlib.pyplot as plt
import sys
import os
import re
import math

def run_eval(phy, packets_per_frame="3", sweep_awgn="0.2,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.9",
             columns="scenario,crc_pass_ratio,goodput_effective_bps,goodput_success_mean_bps,ebn0_db,raw_ber",
             mode="sweep-awgn", **extra_args):
    """指定された PHY で評価を実行し、JSON 結果を返す

    Args:
        phy: PHY の種類 (mary または dsss)
        packets_per_frame: フレームあたりのパケット数
        sweep_awgn: AWGN のスイープ範囲 (カンマ区切り)
        columns: 出力するカラム (カンマ区切り)
        mode: 評価モード
        **extra_args: その他のコマンドライン引数 (例: --some-flag, --key=value)
    """
    cmd = [
        "cargo", "run", "--release", "--bin", "dsss_e2e_eval", "--",
        "--phy", phy,
        "--mode", mode,
        "--packets-per-frame", str(packets_per_frame),
        "--sweep-awgn", str(sweep_awgn),
        "--columns", str(columns),
        "--total-sim-sec", "10",
        "--output", "json"
    ]

    # 追加の引数をコマンドに追加
    for key, value in extra_args.items():
        # key が -- で始まらない場合は追加
        arg_name = key if key.startswith('--') else f'--{key}'
        cmd.append(arg_name)
        # value が True/False/None でない場合は値を文字列として追加
        if value is not True and value is not False and value is not None:
            cmd.append(str(value))

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
    # 実行するケースの設定
    # コマンドライン引数で JSON ファイルを指定することも可能
    cases = [
        {
            "name": "mary",
            "phy": "mary",
            "args": {}
        },
        {
            "name": "dsss",
            "phy": "dsss",
            "args": {}
        }
    ]

    # コマンドライン引数で JSON ファイルが指定されている場合は読み込む
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        try:
            with open(config_file, 'r') as f:
                cases = json.load(f)
            print(f"Loaded {len(cases)} cases from {config_file}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load config file: {e}", file=sys.stderr)
            return 1

    # 各ケースを実行
    results = {}
    for case in cases:
        name = case.get("name", case.get("phy", "unknown"))
        phy = case.get("phy")
        args = case.get("args", {})

        if not phy:
            print(f"Case {name} missing 'phy' field", file=sys.stderr)
            return 1

        print(f"Running case '{name}' with phy={phy}...", file=sys.stderr)
        if args:
            print(f"  Additional args: {args}", file=sys.stderr)

        result = run_eval(phy, **args)
        if result is None:
            print(f"Failed to get results for {name}", file=sys.stderr)
            return 1

        results[name] = result
        print(f"Got {len(result)} data points for {name}", file=sys.stderr)

    # グラフ描画（2x2サブプロット）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    case_names = list(results.keys())
    title_suffix = ' vs '.join(case_names) if len(case_names) <= 3 else f'{len(case_names)} Cases'
    fig.suptitle(f'PHY Performance Comparison: {title_suffix}', fontsize=16)

    # 各サブプロットの設定
    metrics = [
        ('goodput_effective_bps', 'Goodput Effective (bps)', axes[0, 0]),
        ('goodput_success_mean_bps', 'Goodput Success Mean (bps)', axes[0, 1]),
        ('crc_pass_ratio', 'CRC Pass Ratio', axes[1, 0]),
        ('raw_ber', 'Raw BER', axes[1, 1]),
    ]

    # 全データから sigma の最小値・最大値を取得
    all_sigma_values = []
    for case_name in case_names:
        for point in results[case_name]:
            scenario = point.get('scenario', '')
            match = re.search(r'sigma=([\d.]+)', scenario)
            if match:
                sigma = float(match.group(1))
                all_sigma_values.append(sigma)

    if all_sigma_values:
        sigma_min = min(all_sigma_values)
        sigma_max = max(all_sigma_values)
        # dB に変換（ノイズが多いほど dB が大きい）
        db_min = 20 * math.log10(sigma_min)
        db_max = 20 * math.log10(sigma_max)
        # 少し余裕を持たせる
        db_margin = (db_max - db_min) * 0.05
        db_min -= db_margin
        db_max += db_margin
    else:
        db_min, db_max = -20, 0

    for metric, ylabel, ax in metrics:
        # 各ケースのデータを収集
        case_data = {}
        for case_name in case_names:
            data_points = []
            for point in results[case_name]:
                scenario = point.get('scenario', '')
                match = re.search(r'sigma=([\d.]+)', scenario)
                if match:
                    sigma = float(match.group(1))
                    # dB に変換（ノイズが多いほど dB が大きい）
                    db = 20 * math.log10(sigma)
                    metric_val = point.get(metric, 0)
                    data_points.append((db, metric_val))

            # dB でソート
            data_points.sort()
            if data_points:
                x_vals, y_vals = zip(*data_points)
                case_data[case_name] = (x_vals, y_vals)

        # プロット（複数ケースに対応）
        colors = plt.cm.tab10.colors  # 10色のカラーパレット
        for i, (case_name, (x_vals, y_vals)) in enumerate(case_data.items()):
            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, marker='o', label=case_name, linewidth=2, markersize=6, color=color)

        ax.set_xlim(db_min, db_max)
        ax.set_xlabel('Noise Level (dB)', fontsize=11)
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
