import { describe, it, expect, beforeAll } from 'vitest';
import initBase, {
    WasmDsssEncoder as WasmDsssEncoderBase,
    WasmDsssDecoder as WasmDsssDecoderBase,
    WasmMaryEncoder as WasmMaryEncoderBase,
    WasmMaryDecoder as WasmMaryDecoderBase,
    type InitInput,
} from '../pkg/dsp';
import initSimd, {
    WasmDsssEncoder as WasmDsssEncoderSimd,
    WasmDsssDecoder as WasmDsssDecoderSimd,
    WasmMaryEncoder as WasmMaryEncoderSimd,
    WasmMaryDecoder as WasmMaryDecoderSimd,
} from '../pkg-simd/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

type WasmBindings = {
    flavor: 'base' | 'simd';
    wasmPath: string;
    init: (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>) => Promise<unknown>;
    WasmDsssEncoder: new (sampleRate: number, packetsPerBurst: number) => {
        set_data(data: Uint8Array): void;
        pull_frame(): Float32Array | undefined;
    };
    WasmDsssDecoder: new (sampleRate: number, packetsPerBurst: number) => {
        process_samples(samples: Float32Array): { complete: boolean };
        recovered_data(): Uint8Array | undefined;
    };
    WasmMaryEncoder: new (sampleRate: number, packetsPerBurst: number) => {
        set_data(data: Uint8Array): void;
        pull_frame(): Float32Array | undefined;
    };
    WasmMaryDecoder: new (sampleRate: number, packetsPerBurst: number) => {
        process_samples(samples: Float32Array): { complete: boolean };
        recovered_data(): Uint8Array | undefined;
    };
};

const BINDINGS: WasmBindings[] = [
    {
        flavor: 'base',
        wasmPath: path.resolve(__dirname, '../pkg/dsp_bg.wasm'),
        init: initBase,
        WasmDsssEncoder: WasmDsssEncoderBase,
        WasmDsssDecoder: WasmDsssDecoderBase,
        WasmMaryEncoder: WasmMaryEncoderBase,
        WasmMaryDecoder: WasmMaryDecoderBase,
    },
    {
        flavor: 'simd',
        wasmPath: path.resolve(__dirname, '../pkg-simd/dsp_bg.wasm'),
        init: initSimd,
        WasmDsssEncoder: WasmDsssEncoderSimd,
        WasmDsssDecoder: WasmDsssDecoderSimd,
        WasmMaryEncoder: WasmMaryEncoderSimd,
        WasmMaryDecoder: WasmMaryDecoderSimd,
    },
];

async function loadWasm(binding: WasmBindings) {
    const wasmBuffer = fs.readFileSync(binding.wasmPath);
    await binding.init({ module_or_path: wasmBuffer });
}

function trimTrailingZeroBytes(data: Uint8Array): Uint8Array {
    let last = data.length;
    while (last > 0 && data[last - 1] === 0) last--;
    return data.slice(0, last);
}

beforeAll(async () => {
    for (const binding of BINDINGS) {
        await loadWasm(binding);
    }
});

describe('WASM E2E Integration (DSSS Protocol)', () => {
    for (const binding of BINDINGS) {
        it(`should encode and decode through DSSS protocol (${binding.flavor})`, async () => {
            const data = new TextEncoder().encode('Hello DSSS World!');
            const sampleRate = 48000;

            const encoder = new binding.WasmDsssEncoder(sampleRate, 1);
            encoder.set_data(data);

            const decoder = new binding.WasmDsssDecoder(sampleRate, 1);
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

            expect(complete, `[${binding.flavor}] DSSS デコードが完了しませんでした。`).toBe(true);
            const recovered = decoder.recovered_data();
            expect(recovered, `[${binding.flavor}] recovered_data が取得できませんでした。`).toBeTruthy();
            expect(new TextDecoder().decode(trimTrailingZeroBytes(recovered!))).toBe('Hello DSSS World!');
        });
    }
});

describe('WASM E2E Integration (Mary Protocol)', () => {
    for (const binding of BINDINGS) {
        it(`should encode and decode through Mary protocol (${binding.flavor})`, async () => {
            const data = new TextEncoder().encode('Hello Mary World!');
            const sampleRate = 48000;

            const encoder = new binding.WasmMaryEncoder(sampleRate, 3);
            encoder.set_data(data);

            const decoder = new binding.WasmMaryDecoder(sampleRate, 3);
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

            expect(complete, `[${binding.flavor}] Mary デコードが完了しませんでした。`).toBe(true);
            const recovered = decoder.recovered_data();
            expect(recovered, `[${binding.flavor}] recovered_data が取得できませんでした。`).toBeTruthy();
            expect(new TextDecoder().decode(trimTrailingZeroBytes(recovered!))).toBe('Hello Mary World!');
        });
    }
});

describe('Performance Benchmark', () => {
    for (const binding of BINDINGS) {
        it(`should measure real-time processing margin for DSSS at 48kHz (${binding.flavor})`, () => {
            const sampleRate = 48000;
            const data = new TextEncoder().encode('Performance Test Data for DSSS protocol.');
            const encoder = new binding.WasmDsssEncoder(sampleRate, 1);
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

            const decoder = new binding.WasmDsssDecoder(sampleRate, 1);
            const startTime = performance.now();
            decoder.process_samples(allSamples);
            const endTime = performance.now();

            const processingTimeMs = endTime - startTime;
            const audioDurationMs = (allSamples.length / sampleRate) * 1000;
            const margin = audioDurationMs / processingTimeMs;

            console.log(`[Benchmark DSSS][${binding.flavor}]`);
            console.log(`  Audio Duration: ${audioDurationMs.toFixed(2)} ms`);
            console.log(`  Processing Time: ${processingTimeMs.toFixed(2)} ms`);
            console.log(`  Real-time Margin: ${margin.toFixed(2)}x`);

            expect(margin, `[${binding.flavor}] margin`).toBeGreaterThan(1.0);
        }, 20000);

        it(`should measure real-time processing margin for Mary at 48kHz (${binding.flavor})`, () => {
            const sampleRate = 48000;
            const data = new TextEncoder().encode('Performance Test Data for Mary protocol.');
            const encoder = new binding.WasmMaryEncoder(sampleRate, 3);
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

            const decoder = new binding.WasmMaryDecoder(sampleRate, 3);
            const startTime = performance.now();
            decoder.process_samples(allSamples);
            const endTime = performance.now();

            const processingTimeMs = endTime - startTime;
            const audioDurationMs = (allSamples.length / sampleRate) * 1000;
            const margin = audioDurationMs / processingTimeMs;

            console.log(`[Benchmark Mary][${binding.flavor}]`);
            console.log(`  Audio Duration: ${audioDurationMs.toFixed(2)} ms`);
            console.log(`  Processing Time: ${processingTimeMs.toFixed(2)} ms`);
            console.log(`  Real-time Margin: ${margin.toFixed(2)}x`);

            expect(margin, `[${binding.flavor}] margin`).toBeGreaterThan(1.0);
        }, 20000);
    }
});
