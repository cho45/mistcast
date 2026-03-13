#!/usr/bin/env python3
import subprocess
import json
import matplotlib.pyplot as plt
import sys
import os
import math

plt.rcParams["font.family"] = "Noto Sans JP"
plt.rcParams["axes.unicode_minus"] = False

X_AXIS_METRIC = "cn0_db"
X_AXIS_LABEL = "C/N0 (dB-Hz)"

def add_cn0_reference(fig):
    """図の下部に C/N0 の目安テーブルを表示する。"""
    col_labels = ["C/N0", "状態", "例"]
    rows = [
        ["20〜30 dB-Hz", "受信限界", "弱いGNSS"],
        ["30〜40 dB-Hz", "非常に弱い", "屋内GNSS / LoRa(下限)"],
        ["40〜55 dB-Hz", "低SNR通信", "GNSS通常 / DSSS / LoRa(一般)"],
        ["55〜70 dB-Hz", "中品質通信", "衛星通信・放送(低〜中MODCOD)"],
        ["70〜90 dB-Hz", "高品質通信", "FM(高品位) / 衛星TV(高MODCOD)"],
        ["80〜100 dB-Hz", "セルラー広帯域", "LTE(帯域・MCS依存)"],
        ["95〜115 dB-Hz", "超広帯域高速", "Wi-Fi(高MCS)"],
    ]

    ax = fig.add_axes([0.05, 0.02, 0.90, 0.18])
    ax.axis("off")
    ax.text(0.0, 1.05, "C/N0 目安（概算）", ha="left", va="bottom", fontsize=10, transform=ax.transAxes)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.20, 0.58],
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("lightgray")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f1f3f4")

def run_eval(phy, packets_per_frame="2", sweep_awgn="0.2,0.3,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95,1.0,1.2",
             columns="scenario,crc_pass_ratio,goodput_effective_bps,goodput_success_mean_bps,ebn0_db,cn0_db,raw_ber",
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
    required_columns = [
        "scenario",
        X_AXIS_METRIC,
        "goodput_effective_bps",
        "goodput_success_mean_bps",
        "crc_pass_ratio",
        "raw_ber",
    ]
    requested_columns = [col.strip() for col in str(columns).split(",") if col.strip()]
    for col in required_columns:
        if col not in requested_columns:
            requested_columns.append(col)
    merged_columns = ",".join(requested_columns)

    cmd = [
        "cargo", "run", "--release", "--bin", "dsss_e2e_eval", "--",
        "--phy", phy,
        "--mode", mode,
        "--packets-per-frame", str(packets_per_frame),
        "--sweep-awgn", str(sweep_awgn),
        "--columns", merged_columns,
        "--total-sim-sec", "30",
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

    # 全データから x軸メトリクスの最小値・最大値を取得
    all_x_values = []
    for case_name in case_names:
        for point in results[case_name]:
            x_val = point.get(X_AXIS_METRIC)
            if isinstance(x_val, (int, float)) and math.isfinite(x_val):
                all_x_values.append(float(x_val))

    if all_x_values:
        x_min = min(all_x_values)
        x_max = max(all_x_values)
        # 少し余裕を持たせる
        if x_min == x_max:
            x_margin = max(1.0, abs(x_min) * 0.05)
        else:
            x_margin = (x_max - x_min) * 0.05
        x_min -= x_margin
        x_max += x_margin
    else:
        x_min, x_max = 0, 1
        print(f"Warning: no finite '{X_AXIS_METRIC}' values found; using fallback x-axis range.", file=sys.stderr)

    for metric, ylabel, ax in metrics:
        # 各ケースのデータを収集
        case_data = {}
        for case_name in case_names:
            data_points = []
            for point in results[case_name]:
                x_val = point.get(X_AXIS_METRIC)
                metric_val = point.get(metric)
                if isinstance(x_val, (int, float)) and isinstance(metric_val, (int, float)):
                    if math.isfinite(x_val) and math.isfinite(metric_val):
                        data_points.append((float(x_val), float(metric_val)))

            # x軸値でソート
            data_points.sort()
            if data_points:
                x_vals, y_vals = zip(*data_points)
                case_data[case_name] = (x_vals, y_vals)

        # プロット（複数ケースに対応）
        colors = plt.cm.tab10.colors  # 10色のカラーパレット
        for i, (case_name, (x_vals, y_vals)) in enumerate(case_data.items()):
            color = colors[i % len(colors)]
            ax.plot(x_vals, y_vals, marker='o', label=case_name, linewidth=2, markersize=6, color=color)

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(X_AXIS_LABEL, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    add_cn0_reference(fig)
    plt.tight_layout(rect=[0.02, 0.24, 0.995, 0.92])

    # 出力先のディレクトリが存在することを確認
    os.makedirs('docs', exist_ok=True)
    output_path = 'docs/plot_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
