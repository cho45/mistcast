import { describe, it, expect, vi, beforeEach } from 'vitest';

const hoisted = vi.hoisted(() => ({
  expose: vi.fn(),
  init: vi.fn(async () => {}),
  WasmEncoder: vi.fn(),
  WasmDecoder: vi.fn(),
}));

vi.mock('comlink', () => ({
  expose: hoisted.expose,
  transfer: vi.fn((x) => x),
  proxy: vi.fn((x) => x),
}));

vi.mock('../pkg/dsp', () => {
  class WasmEncoder {
    constructor(...args: any[]) {
      hoisted.WasmEncoder(...args);
    }
    set_data() {}
    pull_frame() {
      return new Float32Array(2000);
    }
    encode_all() {
      return new Float32Array(2000);
    }
  }
  class WasmDecoder {
    constructor(...args: any[]) {
      hoisted.WasmDecoder(...args);
    }
    process_samples() {
      return { complete: true, received_packets: 1, needed_packets: 8, progress: 1.0 };
    }
    recovered_data() {
      return new Uint8Array([1, 2, 3]);
    }
    reset() {}
  }
  return {
    default: hoisted.init,
    WasmEncoder,
    WasmDecoder,
  };
});

describe('MistcastBackend', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('initializes WASM only once', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    await backend.init();
    await backend.init();
    expect(hoisted.init).toHaveBeenCalledTimes(1);
  });

  it('starts encoder and appends to audio port', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    const mockPort = {
      postMessage: vi.fn(),
      start: vi.fn(),
    } as unknown as MessagePort;

    await backend.setAudioOutPort(mockPort);
    await backend.startEncoder(new Uint8Array([65, 66]), 48000);

    expect(hoisted.WasmEncoder).toHaveBeenCalledWith(48000);
    // RecycleTransferSender should have been used to postMessage
    expect(mockPort.postMessage).toHaveBeenCalled();
  });

  it('decodes samples and calls onPacket callback', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    const onPacket = vi.fn();

    await backend.startDecoder(10, 8, 48000, onPacket);
    const result = await backend.processSamples(new Float32Array(128));

    expect(hoisted.WasmDecoder).toHaveBeenCalledWith(10, 8, 48000);
    expect(result?.complete).toBe(true);
    expect(onPacket).toHaveBeenCalledWith(new Uint8Array([1, 2, 3]));
  });
});
