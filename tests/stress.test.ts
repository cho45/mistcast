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

        const decoder = new WasmDecoder(sampleRate);

        let complete = false;
        let seed = 0x12345678;
        const nextChunkSize = () => {
            seed = (1664525 * seed + 1013904223) >>> 0;
            return 256 + (seed % 768); // 256..1023
        };

        const processSignal = (signal: Float32Array) => {
            let pos = 0;
            while (pos < signal.length) {
                const chunkSize = nextChunkSize();
                const end = Math.min(pos + chunkSize, signal.length);
                const progress = decoder.process_samples(signal.subarray(pos, end));
                pos = end;
                if (progress.complete) {
                    complete = true;
                    return;
                }
            }
        };

        const silence = new Float32Array(1200);
        for (let i = 0; i < 6 && !complete; i++) {
            const frame = encoder.pull_frame();
            if (!frame) {
                break;
            }

            // 意地悪1: 低ゲイン (1%)
            for (let j = 0; j < frame.length; j++) {
                frame[j] *= 0.01;
            }
            processSignal(frame);
            if (!complete) {
                processSignal(silence);
            }
        }

        expect(complete, "Failed to decode under stress conditions").toBe(true);
    });
});
