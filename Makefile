# 音響通信DSPシステム (outer-kuiper) Makefile

.PHONY: all test test-verbose test-dsp-wasm-simd bench-native-dsp profile-native-dsss profile-wasm-node build example clean check build-wasm dev build-all \
	phy-baseline phy-baseline-full phy-compare \
	dsss-e2e-baseline dsss-e2e-baseline-full dsss-e2e-compare

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

all: build-all

# Rustビルド確認
build:
	cd dsp && cargo build

# WASMビルド
build-wasm:
	npm run build:wasm

# フロントエンドビルド
build-frontend: build-wasm
	npm run build

# 開発サーバー起動
dev:
	npm run dev

# 全ビルド
build-all: build build-wasm build-frontend

# 全テスト実行
test:
	cd dsp && cargo test
	npm test

# 詳細出力でテスト
test-verbose:
	cd dsp && cargo test -- --nocapture
	npm test -- --reporter=verbose

# DSP wasm simd 単体テスト (wasm-bindgen-test)
test-dsp-wasm-simd:
	npm run test:dsp:wasm:simd

bench-native-dsp:
	cd dsp && cargo bench --bench native_dsp

profile-native-dsss:
	bash scripts/profile-native.sh

profile-wasm-node:
	bash scripts/profile-wasm-node.sh
# 個別モジュールのテスト
test-mseq:
	cd dsp && cargo test msequence::tests

test-fec:
	cd dsp && cargo test fec::tests

test-fountain:
	cd dsp && cargo test fountain::tests

test-e2e:
	cd dsp && cargo test e2e -- --nocapture

# encode_wav example実行
example:
	cd dsp && cargo run --example encode_wav -- --input "Hello, acoustic air-gap world!" --output out.wav

# Clippy静的解析
check:
	cd dsp && cargo clippy -- -D warnings

# ビルド成果物削除
clean:
	cd dsp && cargo clean
	rm -f dsp/out.wav

# PHY評価: 現状保存(軽量)
phy-baseline:
	$(PYTHON) scripts/phy_eval/phy_bench.py \
		--out-dir dsp/eval/baselines \
		--trials 40 \
		--min-trials 80 \
		--min-effect 0.20 \
		--payload-bits 256 \
		--deadline-sec 0.8 \
		--max-sec 2.0 \
		--modes sweep-awgn,sweep-cfo,sweep-ppm,sweep-loss,sweep-multipath

# PHY評価: 現状保存(統計重視)
phy-baseline-full:
	$(PYTHON) scripts/phy_eval/phy_bench.py \
		--out-dir dsp/eval/baselines \
		--trials 80 \
		--min-trials 120 \
		--min-effect 0.15 \
		--payload-bits 256 \
		--deadline-sec 0.8 \
		--max-sec 2.0 \
		--modes sweep-awgn,sweep-cfo,sweep-ppm,sweep-loss,sweep-multipath

# 比較: BASE=<metrics.csv> NEW=<metrics.csv> [OUT=<compare.csv>]
phy-compare:
	@test -n "$(BASE)" || (echo "BASE=... を指定してください" && exit 2)
	@test -n "$(NEW)" || (echo "NEW=... を指定してください" && exit 2)
	@ts=$$(date -u +%Y%m%dT%H%M%SZ); \
	out_csv="$(if $(OUT),$(OUT),dsp/eval/runs/compare_$${ts}.csv)"; \
	plot_dir="$(if $(PLOT_DIR),$(PLOT_DIR),dsp/eval/plots/compare_$${ts})"; \
	$(PYTHON) scripts/phy_eval/phy_compare.py \
		--base "$(BASE)" \
		--new "$(NEW)" \
		--out "$$out_csv"; \
	MPLCONFIGDIR="$(CURDIR)/dsp/eval/.mplcache/mplconfig" XDG_CACHE_HOME="$(CURDIR)/dsp/eval/.mplcache/xdg" $(PYTHON) scripts/phy_eval/phy_plot.py \
		--input "$(BASE)" \
		--input "$(NEW)" \
		--label "base" \
		--label "new" \
		--out-dir "$$plot_dir" \
		--metric "$(if $(METRIC),$(METRIC),p_complete_deadline)" \
		--output "compare_overlay_summary.png" \
		--awgn-axis "$(if $(AWGN_AXIS),$(AWGN_AXIS),snr-db)"; \
	echo "compare csv: $$out_csv"; \
	echo "plots dir : $$plot_dir"

# DSSS実経路(E2E)評価: 現状保存(軽量)
dsss-e2e-baseline:
	$(PYTHON) scripts/phy_eval/dsss_e2e_bench.py \
		--out-dir dsp/eval/dsss_e2e/baselines \
		--trials 40 \
		--min-trials 80 \
		--min-effect 0.20 \
		--payload-bytes 64 \
		--deadline-sec 0.8 \
		--max-sec 2.0 \
		--modes sweep-awgn,sweep-ppm,sweep-loss,sweep-fading,sweep-multipath
	@latest_metrics=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_metrics.csv 2>/dev/null | head -n1); \
	latest_limits=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_limits.csv 2>/dev/null | head -n1); \
	latest_meta=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_meta.json 2>/dev/null | head -n1); \
	test -n "$$latest_metrics" && cp "$$latest_metrics" dsp/eval/dsss_e2e/baselines/latest_metrics.csv || true; \
	test -n "$$latest_limits" && cp "$$latest_limits" dsp/eval/dsss_e2e/baselines/latest_limits.csv || true; \
	test -n "$$latest_meta" && cp "$$latest_meta" dsp/eval/dsss_e2e/baselines/latest_meta.json || true

# DSSS実経路(E2E)評価: 現状保存(統計重視)
dsss-e2e-baseline-full:
	$(PYTHON) scripts/phy_eval/dsss_e2e_bench.py \
		--out-dir dsp/eval/dsss_e2e/baselines \
		--trials 80 \
		--min-trials 120 \
		--min-effect 0.15 \
		--payload-bytes 64 \
		--deadline-sec 0.8 \
		--max-sec 2.0 \
		--modes sweep-awgn,sweep-ppm,sweep-loss,sweep-fading,sweep-multipath
	@latest_metrics=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_metrics.csv 2>/dev/null | head -n1); \
	latest_limits=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_limits.csv 2>/dev/null | head -n1); \
	latest_meta=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_meta.json 2>/dev/null | head -n1); \
	test -n "$$latest_metrics" && cp "$$latest_metrics" dsp/eval/dsss_e2e/baselines/latest_metrics.csv || true; \
	test -n "$$latest_limits" && cp "$$latest_limits" dsp/eval/dsss_e2e/baselines/latest_limits.csv || true; \
	test -n "$$latest_meta" && cp "$$latest_meta" dsp/eval/dsss_e2e/baselines/latest_meta.json || true

# DSSS実経路(E2E)比較:
#   既定: NEW=最新 metrics, BASE=その1つ前 metrics
#   明示: BASE=<metrics.csv> NEW=<metrics.csv>
dsss-e2e-compare:
	@ts=$$(date -u +%Y%m%dT%H%M%SZ); \
	history=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_metrics.csv 2>/dev/null || true); \
	new="$(NEW)"; \
	base="$(BASE)"; \
	count=$$(printf '%s\n' "$$history" | sed '/^$$/d' | wc -l | tr -d ' '); \
	if [ -z "$$new" ] && [ -z "$$base" ] && [ "$$count" -eq 1 ]; then \
		base=$$(printf '%s\n' "$$history" | head -n1); \
		echo "baseline が1件のみのため、最新計測を自動実行して比較します"; \
		$(MAKE) --no-print-directory dsss-e2e-baseline; \
		history=$$(ls -1t dsp/eval/dsss_e2e/baselines/20*_metrics.csv 2>/dev/null || true); \
	fi; \
	if [ -z "$$new" ]; then \
		new=$$(printf '%s\n' "$$history" | head -n1); \
	fi; \
	if [ -z "$$base" ]; then \
		base=$$(printf '%s\n' "$$history" | awk -v n="$$new" '$$0 != n { print; exit }'); \
	fi; \
	if [ -z "$$new" ]; then \
		echo "比較対象がありません。先に \`make dsss-e2e-baseline\` を実行してください"; \
		exit 2; \
	fi; \
	if [ -z "$$base" ]; then \
		echo "比較元がありません。最低2回分の baseline metrics が必要です"; \
		echo "例: \`make dsss-e2e-baseline\` をもう一度実行"; \
		exit 2; \
	fi; \
	out_csv="$(if $(OUT),$(OUT),dsp/eval/dsss_e2e/runs/compare_$${ts}.csv)"; \
	plot_dir="$(if $(PLOT_DIR),$(PLOT_DIR),dsp/eval/dsss_e2e/plots/compare_$${ts})"; \
	echo "BASE=$$base"; \
	echo "NEW=$$new"; \
	$(PYTHON) scripts/phy_eval/phy_compare.py \
		--base "$$base" \
		--new "$$new" \
		--out "$$out_csv"; \
	MPLCONFIGDIR="$(CURDIR)/dsp/eval/.mplcache/mplconfig" XDG_CACHE_HOME="$(CURDIR)/dsp/eval/.mplcache/xdg" $(PYTHON) scripts/phy_eval/phy_plot.py \
		--input "$$base" \
		--input "$$new" \
		--label "base" \
		--label "new" \
		--out-dir "$$plot_dir" \
		--metric "$(if $(METRIC),$(METRIC),p_complete_deadline)" \
		--output "compare_overlay_summary.png" \
		--awgn-axis "$(if $(AWGN_AXIS),$(AWGN_AXIS),snr-db)"; \
	echo "compare csv: $$out_csv"; \
	echo "plots dir : $$plot_dir"
