import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('AudioStreamProcessor', () => {
  beforeEach(() => {
    vi.stubGlobal('sampleRate', 48000);
    vi.stubGlobal('currentTime', 0);
    // Mock AudioWorkletProcessor
    vi.stubGlobal('AudioWorkletProcessor', class {
        port = {
            onmessage: null as any,
            postMessage: vi.fn(),
        };
    });
  });

  it('registers itself', async () => {
    const registerMock = vi.fn();
    vi.stubGlobal('registerProcessor', registerMock);

    await import('./audio-stream-processor');
    expect(registerMock).toHaveBeenCalledWith('audio-stream-processor', expect.any(Function));
  });

  it('processes audio and drains packet queue', async () => {
    const registerMock = vi.fn();
    vi.stubGlobal('registerProcessor', registerMock);
    const { AudioStreamProcessor } = await import('./audio-stream-processor');
    
    const processor = new AudioStreamProcessor();
    (processor as any).baseMinStartSamples = 0; 
    (processor as any).baseLowWaterSamples = 0;
    (processor as any).recomputeBufferTargets();
    
    const outputL = new Float32Array(128);
    const outputR = new Float32Array(128);
    const outputs = [[outputL, outputR]];
    const inputs: Float32Array[][] = [[new Float32Array(128)]];

    // Initially silent
    processor.process(inputs, outputs);
    expect(outputL[0]).toBe(0);

    // Push data
    const inputPort = {
        postMessage: vi.fn(),
        start: vi.fn(),
    };
    (processor as any).port.onmessage({ data: { type: 'attach-input-port', port: inputPort } });
    
    const testBuffer = new Float32Array(5000).fill(0.5);
    (processor as any).handleInputMessage({ type: 'push', data: testBuffer });

    processor.process(inputs, outputs);
    expect(outputL[0]).toBe(0.5);
  });
});
