import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmDsssEncoder, WasmDsssDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init(wasmBuffer);
}

describe('WASM E2E Integration (Fixed Protocol)', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should encode and decode through fixed K=10 protocol', async () => {
        const data = new TextEncoder().encode("Hello Acoustic World!");
        const sampleRate = 48000;
        
        const encoder = new WasmDsssEncoder(sampleRate);
        encoder.set_data(data);
        
        const decoder = new WasmDsssDecoder(sampleRate);
        let complete = false;
        
        for (let i = 0; i < 30; i++) {
            const frame = encoder.pull_frame();
            if (!frame) break;
            
            const signal = new Float32Array(frame.length + 4800);
            signal.set(frame);
            
            const progress = decoder.process_samples(signal);
            if (progress.complete) {
                complete = true;
                break;
            }
        }

        expect(complete, "デコードが完了しませんでした。").toBe(true);
        const recovered = decoder.recovered_data();
        expect(recovered).toBeDefined();
        
        let last = recovered!.length;
        while(last > 0 && recovered![last-1] === 0) last--;
        expect(new TextDecoder().decode(recovered!.slice(0, last))).toBe("Hello Acoustic World!");
    });
});

describe('Performance Benchmark (Fixed Protocol)', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should measure real-time processing margin at 48kHz', () => {
        const sampleRate = 48000;
        const data = new TextEncoder().encode("Performance Test Data for FIXED_K=10 protocol.");
        const encoder = new WasmDsssEncoder(sampleRate);
        encoder.set_data(data);

        let allSamples = new Float32Array(0);
        for (let i = 0; i < 15; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const next = new Float32Array(allSamples.length + frame.length + 4800);
                next.set(allSamples);
                next.set(frame, allSamples.length);
                allSamples = next;
            }
        }
        
        const decoder = new WasmDsssDecoder(sampleRate);
        const startTime = performance.now();
        decoder.process_samples(allSamples);
        const endTime = performance.now();
        
        const processingTimeMs = endTime - startTime;
        const audioDurationMs = (allSamples.length / sampleRate) * 1000;
        const margin = audioDurationMs / processingTimeMs;
        
        console.log(`[Benchmark]`);
        console.log(`  Audio Duration: ${audioDurationMs.toFixed(2)} ms`);
        console.log(`  Processing Time: ${processingTimeMs.toFixed(2)} ms`);
        console.log(`  Real-time Margin: ${margin.toFixed(2)}x`);
        
        expect(margin).toBeGreaterThan(1.0);
    });
});
