# 音響通信DSPシステム (outer-kuiper) Makefile

.PHONY: all test test-verbose build example clean check

all: build

# ビルド確認
build:
	cd dsp && cargo build

# 全テスト実行
test:
	cd dsp && cargo test

# 詳細出力でテスト
test-verbose:
	cd dsp && cargo test -- --nocapture

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
