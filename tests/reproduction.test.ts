import { describe, it, expect, beforeAll } from 'vitest';
import init, { WasmDsssEncoder, WasmDsssDecoder } from '../pkg/dsp';
import * as fs from 'node:fs';
import * as path from 'node:path';

async function loadWasm() {
    const wasmPath = path.resolve(__dirname, '../pkg/dsp_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);
    await init({ module_or_path: wasmBuffer });
}

describe('Reproduction: 10/10 Verification', () => {
    beforeAll(async () => {
        await loadWasm();
    });

    it('should decode with warmup and finish within 10 packets', async () => {
        const data = new Uint8Array(16);
        for(let i=0; i<16; i++) data[i] = 65 + i; // "ABC..."
        
        const sampleRate = 48000;
        const encoder = new WasmDsssEncoder(sampleRate, 1);
        encoder.set_data(data);
        
        const decoder = new WasmDsssDecoder(sampleRate, 1);

        // 受信系の初期過渡を避けるために事前に無音を流す。
        decoder.process_samples(new Float32Array(4096));

        let completedAt = -1;
        for (let i = 0; i < 10; i++) {
            const frame = encoder.pull_frame();
            if (frame) {
                const signal = new Float32Array(frame.length + 4800);
                signal.set(frame);
                const progress = decoder.process_samples(signal);
                console.log(
                    `Packet ${i}: Received=${progress.received_packets} Rank=${progress.rank_packets} Progress=${progress.progress}`
                );
                if (progress.complete) {
                    completedAt = i;
                    break;
                }
            }
        }

        expect(completedAt, "10 packets以内で復号完了すべき").toBeGreaterThanOrEqual(0);
        const recovered = decoder.recovered_data();
        expect(recovered?.slice(0, 16)).toEqual(data);
    });

    it('should recover quickly even when part of early packets are dropped', async () => {
        // k=10 を強制するため 160 bytes 送る。
        const data = new Uint8Array(160);
        for (let i = 0; i < data.length; i++) data[i] = i & 0xff;

        const sampleRate = 48000;
        const encoder = new WasmDsssEncoder(sampleRate, 1);
        encoder.set_data(data);
        const decoder = new WasmDsssDecoder(sampleRate, 1);
        decoder.process_samples(new Float32Array(4096));

        let complete = false;
        const seenRanks: number[] = [];

        for (let i = 0; i < 40; i++) {
            const frame = encoder.pull_frame();
            if (!frame) break;

            // seq 5..9 に相当するフレームを意図的にドロップ。
            if (i >= 5 && i <= 9) {
                const silence = encoder.modulate_silence(4800);
                decoder.process_samples(silence);
                continue;
            }

            const silence = encoder.modulate_silence(4800);
            const signal = new Float32Array(frame.length + silence.length);
            signal.set(frame);
            signal.set(silence, frame.length);
            const progress = decoder.process_samples(signal);
            seenRanks.push(progress.rank_packets);
            console.log(
                `drop-test i=${i}: recv=${progress.received_packets} rank=${progress.rank_packets} progress=${progress.progress}`
            );
            if (progress.complete) {
                complete = true;
                break;
            }
        }

        const tails = encoder.flush();
        decoder.process_samples(tails);

        // バースト化で rank が複数ずつ進むことがあるため、完了までの反復数で評価する。
        // パケット数が PPB=1 に減少したため、完了までの反復数上限を 6 から 12 に調整する。
        expect(seenRanks.length, "回復までの反復が長すぎる").toBeLessThanOrEqual(12);
        expect(
            seenRanks.every((r, i) => i === 0 || r >= seenRanks[i - 1]),
            "rank は単調非減少であるべき"
        ).toBe(true);
        expect(complete, "停滞後に回復して完了すべき").toBe(true);
        const recovered = decoder.recovered_data();
        expect(recovered?.slice(0, data.length)).toEqual(data);
    });

    it('should recover for k=2 even when first two packets are missed', async () => {
        // PAYLOAD_SIZE=24 のため、25 bytes で k=2 を強制する。
        const data = new Uint8Array(25);
        for (let i = 0; i < data.length; i++) data[i] = 65 + (i % 26);

        const sampleRate = 48000;
        const encoder = new WasmDsssEncoder(sampleRate, 1);
        encoder.set_data(data);
        const decoder = new WasmDsssDecoder(sampleRate, 1);
        decoder.process_samples(new Float32Array(4096));

        const seenRanks: number[] = [];
        let last: ReturnType<typeof decoder.process_samples> | null = null;
        let complete = false;

        for (let i = 0; i < 24; i++) {
            const frame = encoder.pull_frame();
            if (!frame) break;

            // seq=0,1 の先頭2パケットを意図的に欠落させる。
            if (i < 2) continue;

            const signal = new Float32Array(frame.length + 4800);
            signal.set(frame);
            const progress = decoder.process_samples(signal);
            last = progress;
            seenRanks.push(progress.rank_packets);
            console.log(
                `k2-stall i=${i}: recv=${progress.received_packets} rank=${progress.rank_packets}/${progress.needed_packets} dep=${progress.dependent_packets} dup=${progress.duplicate_packets} complete=${progress.complete}`
            );
            if (progress.complete) {
                complete = true;
                break;
            }
        }

        expect(last).not.toBeNull();
        expect(last!.needed_packets).toBe(2);
        expect(complete, "k=2 で先頭2パケット欠落でも完了すべき").toBe(true);
        expect(Math.max(...seenRanks), "rank=2 まで到達するはず").toBe(2);
    });

    describe('Size prefix encoding/decoding', () => {
        it('should encode and decode data with size prefix correctly', async () => {
            const originalData = new Uint8Array([65, 66, 67, 68, 69]); // "ABCDE"
            const sampleRate = 48000;

            // エンコーダ側: サイズプレフィックスを追加
            const sizeBuffer = new ArrayBuffer(2);
            const sizeView = new DataView(sizeBuffer);
            sizeView.setUint16(0, originalData.length, false); // ビッグエンディアン
            const sizePrefix = new Uint8Array(sizeBuffer);

            const dataWithSize = new Uint8Array(2 + originalData.length);
            dataWithSize.set(sizePrefix, 0);
            dataWithSize.set(originalData, 2);

            // WASMエンコーダ/デコーダで処理
            const encoder = new WasmDsssEncoder(sampleRate, 1);
            encoder.set_data(dataWithSize);
            const decoder = new WasmDsssDecoder(sampleRate, 1);
            decoder.process_samples(new Float32Array(4096));

            // パケットを処理
            for (let i = 0; i < 20; i++) {
                const frame = encoder.pull_frame();
                if (!frame) break;
                const signal = new Float32Array(frame.length + 4800);
                signal.set(frame);
                const progress = decoder.process_samples(signal);
                if (progress.complete) break;
            }

            const recovered = decoder.recovered_data();
            expect(recovered).toBeDefined();

            // デコーダ側: サイズプレフィックスから元のサイズを読み取り
            const recoveredSizeView = new DataView(recovered!.buffer, recovered!.byteOffset, 2);
            const originalSize = recoveredSizeView.getUint16(0, false); // ビッグエンディアン

            // サイズプレフィックスを除いた実際のデータを切り出し
            const actualData = recovered!.slice(2, 2 + originalSize);

            expect(originalSize).toBe(originalData.length);
            expect(actualData).toEqual(originalData);
        });

        it('should correctly recover data with trailing zeros (WEBP-like)', async () => {
            // WEBPファイルのような、末尾が 00 00 のデータ
            const webpLikeData = new Uint8Array([
                0x52, 0x49, 0x46, 0x46, // "RIFF"
                0x1A, 0x0F, 0x00, 0x00, // filesize (little endian)
                0x57, 0x45, 0x42, 0x50, // "WEBP"
                0x00, 0x00, // 末尾のゼロ（重要）
            ]);

            const sampleRate = 48000;

            // エンコーダ側: サイズプレフィックスを追加
            const sizeBuffer = new ArrayBuffer(2);
            const sizeView = new DataView(sizeBuffer);
            sizeView.setUint16(0, webpLikeData.length, false);
            const sizePrefix = new Uint8Array(sizeBuffer);

            const dataWithSize = new Uint8Array(2 + webpLikeData.length);
            dataWithSize.set(sizePrefix, 0);
            dataWithSize.set(webpLikeData, 2);

            // WASMエンコーダ/デコーダで処理
            const encoder = new WasmDsssEncoder(sampleRate, 1);
            encoder.set_data(dataWithSize);
            const decoder = new WasmDsssDecoder(sampleRate, 1);
            decoder.process_samples(new Float32Array(4096));

            // パケットを処理
            for (let i = 0; i < 20; i++) {
                const frame = encoder.pull_frame();
                if (!frame) break;
                const signal = new Float32Array(frame.length + 4800);
                signal.set(frame);
                const progress = decoder.process_samples(signal);
                if (progress.complete) break;
            }

            const recovered = decoder.recovered_data();
            expect(recovered).toBeDefined();

            // デコーダ側: サイズプレフィックスから元のサイズを読み取り
            const recoveredSizeView = new DataView(recovered!.buffer, recovered!.byteOffset, 2);
            const originalSize = recoveredSizeView.getUint16(0, false);

            // サイズプレフィックスを除いた実際のデータを切り出し
            const actualData = recovered!.slice(2, 2 + originalSize);

            // 末尾の 00 00 が保持されていることを確認
            expect(actualData.length).toBe(webpLikeData.length);
            expect(actualData[actualData.length - 2]).toBe(0x00);
            expect(actualData[actualData.length - 1]).toBe(0x00);
            expect(actualData).toEqual(webpLikeData);
        });

        it('should handle various data sizes with size prefix', async () => {
            const testCases = [
                new Uint8Array([]), // 空データ
                new Uint8Array([42]), // 1バイト
                new Uint8Array([1, 2, 3, 4, 5]), // 5バイト
                new Uint8Array(Array.from({ length: 100 }, (_, i) => i & 0xff)), // 100バイト
                new Uint8Array(Array.from({ length: 255 }, (_, i) => i & 0xff)), // 255バイト（u16最大）
            ];

            for (const originalData of testCases) {
                const sampleRate = 48000;

                // エンコーダ側: サイズプレフィックスを追加
                const sizeBuffer = new ArrayBuffer(2);
                const sizeView = new DataView(sizeBuffer);
                sizeView.setUint16(0, originalData.length, false);
                const sizePrefix = new Uint8Array(sizeBuffer);

                const dataWithSize = new Uint8Array(2 + originalData.length);
                dataWithSize.set(sizePrefix, 0);
                dataWithSize.set(originalData, 2);

                // WASMエンコーダ/デコーダで処理
                const encoder = new WasmDsssEncoder(sampleRate, 1);
                encoder.set_data(dataWithSize);
                const decoder = new WasmDsssDecoder(sampleRate, 1);
                decoder.process_samples(new Float32Array(4096));

                // パケットを処理
                for (let i = 0; i < 50; i++) {
                    const frame = encoder.pull_frame();
                    if (!frame) break;
                    const signal = new Float32Array(frame.length + 4800);
                    signal.set(frame);
                    const progress = decoder.process_samples(signal);
                    if (progress.complete) break;
                }

                const recovered = decoder.recovered_data();
                expect(recovered).toBeDefined();

                // デコーダ側: サイズプレフィックスから元のサイズを読み取り
                const recoveredSizeView = new DataView(recovered!.buffer, recovered!.byteOffset, 2);
                const originalSize = recoveredSizeView.getUint16(0, false);

                // サイズプレフィックスを除いた実際のデータを切り出し
                const actualData = recovered!.slice(2, 2 + originalSize);

                expect(originalSize).toBe(originalData.length);
                expect(actualData).toEqual(originalData);
            }
        });
    });
});
