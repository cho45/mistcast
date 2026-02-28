import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmEncoder, WasmDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

// Node.js環境でWASMファイルをロードするためのヘルパー
async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init(wasmBuffer);
}

describe('WASM E2E Integration', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should encode and decode data through WASM interface', async () => {
        const data = new TextEncoder().encode("WASM Acoustic Test");
        const sampleRate = 48000;
        
        const encoder = new WasmEncoder(sampleRate);
        const samples = encoder.encode_all(data);
        
        expect(samples.length).toBeGreaterThan(0);

        const decoder = new WasmDecoder(data.length, 8, sampleRate);
        
        // チャンクに分けて処理
        const chunkSize = 2048;
        let complete = false;
        
        for (let i = 0; i < samples.length; i += chunkSize) {
            const chunk = samples.slice(i, i + chunkSize);
            const progress = decoder.process_samples(chunk);
            if (progress.complete) {
                complete = true;
                break;
            }
        }

        expect(complete).toBe(true);
        const recovered = decoder.recovered_data();
        expect(recovered).toBeDefined();
        expect(new TextDecoder().decode(recovered)).toBe("WASM Acoustic Test");
    });
});

describe('Performance Benchmark', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should measure real-time processing margin at 48kHz', () => {
        const sampleRate = 48000;
        const data = new TextEncoder().encode("Performance Test Data for Real-time Margin Calculation.");
        const encoder = new WasmEncoder(sampleRate);
        const samples = encoder.encode_all(data);
        
        const decoder = new WasmDecoder(data.length, 8, sampleRate);
        
        const startTime = performance.now();
        
        // 全サンプルを一気に処理（内部でループ処理される）
        decoder.process_samples(samples);
        
        const endTime = performance.now();
        const processingTimeMs = endTime - startTime;
        const audioDurationMs = (samples.length / sampleRate) * 1000;
        
        const margin = audioDurationMs / processingTimeMs;
        
        console.log(`[Benchmark]`);
        console.log(`  Audio Duration: ${audioDurationMs.toFixed(2)} ms`);
        console.log(`  Processing Time: ${processingTimeMs.toFixed(2)} ms`);
        console.log(`  Real-time Margin: ${margin.toFixed(2)}x`);
        
        expect(margin).toBeGreaterThan(1.0); // 1.0未満だとリアルタイム処理が不可能
    });
});
