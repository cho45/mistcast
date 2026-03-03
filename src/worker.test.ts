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
  class WasmDsssEncoder {
    constructor(...args: any[]) {
      hoisted.WasmEncoder(...args);
    }
    set_data = vi.fn();
    pull_frame = vi.fn(() => new Float32Array(2000));
    reset = vi.fn();
  }
  class WasmDsssDecoder {
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
      progress: 0.1,
      basis_matrix: new Uint8Array(121)
    }));
    recovered_data = vi.fn(() => new Uint8Array([0, 3, 1, 2, 3]));
    reset = vi.fn();
  }
  return {
    default: hoisted.initBase,
    WasmDsssEncoder,
    WasmDsssDecoder,
    WasmMaryEncoder: WasmDsssEncoder,
    WasmMaryDecoder: WasmDsssDecoder,
  };
});

vi.mock('../pkg-simd/dsp', () => {
  class WasmDsssEncoder {
    constructor(...args: any[]) {
      hoisted.WasmEncoder(...args);
    }
    set_data = vi.fn();
    pull_frame = vi.fn(() => new Float32Array(2000));
    reset = vi.fn();
  }
  class WasmDsssDecoder {
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
      progress: 0.1,
      basis_matrix: new Uint8Array(121)
    }));
    recovered_data = vi.fn(() => new Uint8Array([0, 3, 1, 2, 3]));
    reset = vi.fn();
  }
  return {
    default: hoisted.initSimd,
    WasmDsssEncoder,
    WasmDsssDecoder,
    WasmMaryEncoder: WasmDsssEncoder,
    WasmMaryDecoder: WasmDsssDecoder,
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

  it('decodes samples and extracts actual data from size prefix', async () => {
    const { MistcastBackend } = await import('./worker');
    const backend = new MistcastBackend();
    const onPacket = vi.fn();
    const onProgress = vi.fn();

    await backend.startDecoder(48000, onPacket, onProgress);
    const result = await backend.processSamples(new Float32Array(128));

    expect(hoisted.WasmDecoder).toHaveBeenCalledWith(48000);
    expect(result?.complete).toBe(true);

    // recovered_data は [size: 2bytes][actual_data] の形式を返す
    // モックでは [1, 2, 3] を返しているので、サイズプレフィックスを考慮すると:
    // - 先頭2バイト: 0x00, 0x03 (size=3, big endian)
    // - 実データ: [1, 2, 3]
    // テストでは簡易的に [0, 3, 1, 2, 3] を返すようにモックを修正
    const expectedSizePrefix = new Uint8Array([0, 3]); // size=3
    const expectedData = new Uint8Array([1, 2, 3]);
    const expectedWithPrefix = new Uint8Array([...expectedSizePrefix, ...expectedData]);

    // onPacket にはサイズプレフィックスを除いた実際のデータが渡される
    expect(onPacket).toHaveBeenCalledWith(expectedData);
    expect(onProgress).toHaveBeenCalled();
  });
});
