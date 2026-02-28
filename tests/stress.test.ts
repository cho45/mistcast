import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmEncoder, WasmDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init(wasmBuffer);
}

describe('WASM Reality Stress Test (Fixed Protocol)', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should handle small chunks and low gain', async () => {
        const data = new TextEncoder().encode("Hello Acoustic World!");
        const sampleRate = 48000;
        
        const encoder = new WasmEncoder(sampleRate);
        encoder.set_data(data);
        
        let fullSignal = new Float32Array(0);
        for (let i = 0; i < 20; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const next = new Float32Array(fullSignal.length + frame.length + 4800);
                next.set(fullSignal);
                next.set(frame, fullSignal.length);
                fullSignal = next;
            }
        }

        const decoder = new WasmDecoder(sampleRate);
        
        // 意地悪1: 低ゲイン (1%)
        for(let i=0; i<fullSignal.length; i++) fullSignal[i] *= 0.01;

        let complete = false;
        let pos = 0;
        while (pos < fullSignal.length) {
            // 意地悪2: 不規則なチャンクサイズ (128〜512)
            const chunkSize = 128 + Math.floor(Math.random() * 384);
            const chunk = fullSignal.slice(pos, pos + chunkSize);
            pos += chunkSize;

            const progress = decoder.process_samples(chunk);
            if (progress.complete) {
                complete = true;
                break;
            }
        }

        expect(complete, "Failed to decode under stress conditions").toBe(true);
    });
});
