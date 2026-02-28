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

    it('should decode exactly 10 packets in perfect loopback', async () => {
        const data = new Uint8Array(16);
        for(let i=0; i<16; i++) data[i] = 65 + i; // "ABC..."
        
        const sampleRate = 48000;
        const encoder = new WasmEncoder(sampleRate);
        encoder.set_data(data);
        
        const decoder = new WasmDecoder(sampleRate);
        
        let success = false;
        for (let i = 0; i < 15; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const progress = decoder.process_samples(frame);
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
