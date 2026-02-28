import { describe, it, expect, vi, beforeEach } from 'vitest';

const hoisted = vi.hoisted(() => ({
  expose: vi.fn(),
  initBase: vi.fn(async () => {}),
  initSimd: vi.fn(async () => {}),
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
    set_data = vi.fn();
    get_k = vi.fn(() => 10);
    pull_frame = vi.fn(() => new Float32Array(2000));
    encode_all = vi.fn(() => new Float32Array(2000));
  }
  class WasmDecoder {
    constructor(...args: any[]) {
      hoisted.WasmDecoder(...args);
    }
    process_samples = vi.fn(() => ({
      complete: true,
      received_packets: 1,
      needed_packets: 11,
      rank_packets: 1,
      stalled_packets: 0,
      dependent_packets: 0,
      duplicate_packets: 0,
      crc_error_packets: 0,
      parse_error_packets: 0,
      invalid_neighbor_packets: 0,
      last_packet_seq: 0,
      last_rank_up_seq: 0,
      progress: 0.1
    }));
    recovered_data = vi.fn(() => new Uint8Array([1, 2, 3]));
    reset = vi.fn();
  }
  return {
    default: hoisted.initBase,
    WasmEncoder,
    WasmDecoder,
  };
});

vi.mock('../pkg-simd/dsp', () => {
  class WasmEncoder {
    constructor(...args: any[]) {
      hoisted.WasmEncoder(...args);
    }
    set_data = vi.fn();
    get_k = vi.fn(() => 10);
    pull_frame = vi.fn(() => new Float32Array(2000));
    encode_all = vi.fn(() => new Float32Array(2000));
  }
  class WasmDecoder {
    constructor(...args: any[]) {
      hoisted.WasmDecoder(...args);
    }
    process_samples = vi.fn(() => ({
      complete: true,
      received_packets: 1,
      needed_packets: 11,
      rank_packets: 1,
      stalled_packets: 0,
      dependent_packets: 0,
      duplicate_packets: 0,
      crc_error_packets: 0,
      parse_error_packets: 0,
      invalid_neighbor_packets: 0,
      last_packet_seq: 0,
      last_rank_up_seq: 0,
      progress: 0.1
    }));
    recovered_data = vi.fn(() => new Uint8Array([1, 2, 3]));
    reset = vi.fn();
  }
  return {
    default: hoisted.initSimd,
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
    expect(hoisted.initBase.mock.calls.length + hoisted.initSimd.mock.calls.length).toBe(1);
  });

  it('starts encoder and appends to audio port', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    const mockPort = {
      postMessage: vi.fn(),
      start: vi.fn(),
    } as unknown as MessagePort;

    await backend.setAudioOutPort(mockPort);
    // startEncoder does not return anything in the latest implementation
    await backend.startEncoder(new Uint8Array([65, 66]), 48000);

    expect(hoisted.WasmEncoder).toHaveBeenCalledWith(48000);
    expect(mockPort.postMessage).toHaveBeenCalled();
  });

  it('decodes samples and calls onPacket callback', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    const onPacket = vi.fn();
    const onProgress = vi.fn();

    await backend.startDecoder(48000, onPacket, onProgress);
    const result = await backend.processSamples(new Float32Array(128));

    expect(hoisted.WasmDecoder).toHaveBeenCalledWith(48000);
    expect(result?.complete).toBe(true);
    expect(onPacket).toHaveBeenCalledWith(new Uint8Array([1, 2, 3]));
    expect(onProgress).toHaveBeenCalled();
  });
});
