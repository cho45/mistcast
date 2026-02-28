# 音響通信DSPシステム (outer-kuiper) Makefile

.PHONY: all test test-verbose build example clean check build-wasm dev build-all

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
