import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmDsssEncoder, WasmDsssDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init(wasmBuffer);
}

describe('Reliability Stress Test (Fixed Protocol)', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should succeed 100% of the time over 10 repeated runs', async () => {
        const sampleRate = 48000;
        const data = new TextEncoder().encode("Reliability Test");
        
        for (let run = 1; run <= 10; run++) {
            const encoder = new WasmDsssEncoder(sampleRate, 1);
            encoder.set_data(data);

            const decoder = new WasmDsssDecoder(sampleRate, 1);
            let complete = false;
            let p_count = 0;
            
            // FIXED_K=10 なので、最低11パケット必要。余裕を持って20パケット送信。
            while (!complete && p_count < 20) {
                const frame = encoder.pull_frame();
                if (!frame) break;
                p_count++;

                const signal = new Float32Array(frame.length + 4800);
                signal.set(frame);
                
                const progress = decoder.process_samples(signal);
                if (progress.complete) {
                    complete = true;
                }
            }

            expect(complete, `Run ${run} failed.`).toBe(true);
        }
    }, 20000);

    it('should succeed even with dirty buffers by ensuring full decoder reset', async () => {
        const sampleRate = 48000;
        const data = new TextEncoder().encode("Dirty Restart");
        const encoder = new WasmDsssEncoder(sampleRate, 1);

        // 1. ゴミを流す
        encoder.set_data(new TextEncoder().encode("Garbage"));
        const trashDecoder = new WasmDsssDecoder(sampleRate, 1);
        for(let i=0; i<3; i++) {
            const f = encoder.pull_frame();
            if (f) trashDecoder.process_samples(f);
        }

        // 2. 本番（デコーダを新調）
        const freshDecoder = new WasmDsssDecoder(sampleRate, 1);
        encoder.set_data(data);
        
        let complete = false;
        for(let i=0; i<20; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const progress = freshDecoder.process_samples(frame);
                if (progress.complete) {
                    complete = true;
                    break;
                }
            }
        }
        
        expect(complete, "Should succeed with a fresh decoder").toBe(true);
    });
});
