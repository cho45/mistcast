import { describe, it, expect, vi, beforeEach } from 'vitest';

// AudioWorkletの環境を可能な限りNode.jsで模倣する
class MockAudioWorkletProcessor {
    port = {
        onmessage: null as any,
        postMessage: vi.fn(),
    };
}

describe('DecoderProcessor Logic Verification', () => {
    it('should NOT drop samples when port is attached asynchronously', async () => {
        // audio-processors.ts の実装をエミュレート
        let outputPort: any = null;
        const CHUNK_SIZE = 1024;
        let buffer = new Float32Array(CHUNK_SIZE);
        let pos = 0;

        const process = (input: Float32Array) => {
            if (!outputPort) return; // ← ここで落ちているのではないか？
            for (let i = 0; i < input.length; i++) {
                buffer[pos++] = input[i];
                if (pos >= CHUNK_SIZE) {
                    outputPort.postMessage({ type: 'input', data: buffer });
                    pos = 0;
                }
            }
        };

        const inL = new Float32Array(128).fill(0.5);
        
        // 1. 最初はポートがない状態で process が呼ばれる (AudioContextが開始される)
        process(inL);
        expect(pos).toBe(0); // outputPort が null なので何も蓄積されていない！ (致命的なデータ損失)

        // 2. 後からポートがアタッチされる
        outputPort = { postMessage: vi.fn() };
        process(inL);
        expect(pos).toBe(128); // ここでようやく蓄積が始まるが、最初のデータは永遠に失われた
    });
});
