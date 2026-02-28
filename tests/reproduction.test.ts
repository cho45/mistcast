import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmEncoder, WasmDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init(wasmBuffer);
}

describe('Reproduction: 10/10 Verification', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should decode in perfect loopback with deterministic framing', async () => {
        const data = new Uint8Array(16);
        for(let i=0; i<16; i++) data[i] = 65 + i; // "ABC..."
        
        const sampleRate = 48000;
        const encoder = new WasmEncoder(sampleRate);
        encoder.set_data(data);
        
        const decoder = new WasmDecoder(sampleRate);
        
        let success = false;
        // 固定プロトコルでは同一条件の他テストと揃えて
        // 「フレーム + 無音ギャップ」を流す。
        for (let i = 0; i < 30; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const signal = new Float32Array(frame.length + 4800);
                signal.set(frame);
                const progress = decoder.process_samples(signal);
                console.log(`Packet ${i}: Received=${progress.received_packets}, Progress=${progress.progress}`);
                if (progress.complete) {
                    success = true;
                    break;
                }
            }
        }

        expect(success).toBe(true);
        const recovered = decoder.recovered_data();
        expect(recovered?.slice(0, 16)).toEqual(data);
    });
});
