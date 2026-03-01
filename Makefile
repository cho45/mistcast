# 音響通信DSPシステム (outer-kuiper) Makefile

.PHONY: all test test-verbose build example clean check build-wasm dev build-all \
	phy-baseline phy-baseline-full phy-compare phy-compare-threeway phy-plot

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
	$(PYTHON) scripts/phy_eval/phy_compare.py \
		--base "$(BASE)" \
		--new "$(NEW)" \
		--out "$(if $(OUT),$(OUT),dsp/eval/runs/compare_$(shell date -u +%Y%m%dT%H%M%SZ).csv)"

# DSSS改善比較: BASE=<baseline_metrics.csv> NEW=<new_metrics.csv> [OUT=<compare.csv>] [ALPHA=0.05]
phy-compare-threeway:
	@test -n "$(BASE)" || (echo "BASE=... を指定してください" && exit 2)
	@test -n "$(NEW)" || (echo "NEW=... を指定してください" && exit 2)
	$(PYTHON) scripts/phy_eval/phy_compare_threeway.py \
		--base "$(BASE)" \
		--new "$(NEW)" \
		--alpha "$(if $(ALPHA),$(ALPHA),0.05)" \
		--out "$(if $(OUT),$(OUT),dsp/eval/runs/threeway_$(shell date -u +%Y%m%dT%H%M%SZ).csv)"

# 可視化: INPUT=<metrics.csv> [OUT_DIR=<dir>] [METRIC=p_complete_deadline] [OUTPUT=phy_summary.png]
phy-plot:
	@test -n "$(INPUT)" || (echo "INPUT=... を指定してください" && exit 2)
	MPLCONFIGDIR="$(CURDIR)/dsp/eval/.mplcache/mplconfig" XDG_CACHE_HOME="$(CURDIR)/dsp/eval/.mplcache/xdg" $(PYTHON) scripts/phy_eval/phy_plot.py \
		--input "$(INPUT)" \
		--out-dir "$(if $(OUT_DIR),$(OUT_DIR),dsp/eval/plots/$(shell date -u +%Y%m%dT%H%M%SZ))" \
		--metric "$(if $(METRIC),$(METRIC),p_complete_deadline)" \
		--output "$(if $(OUTPUT),$(OUTPUT),phy_summary.png)"
