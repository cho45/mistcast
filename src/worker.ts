import { expose } from "comlink";
import initBase, { WasmEncoder as WasmEncoderBase, WasmDecoder as WasmDecoderBase } from "../pkg/dsp";
import initSimd, { WasmEncoder as WasmEncoderSimd, WasmDecoder as WasmDecoderSimd } from "../pkg-simd/dsp";
import { RecycleTransferSender } from "./recycle-transfer-bridge";

type WasmEncoderLike = {
  set_data(data: Uint8Array): void;
  pull_frame(): Float32Array | null | undefined;
};

type WasmDecoderLike = {
  process_samples(samples: Float32Array): {
    received_packets: number;
    needed_packets: number;
    rank_packets: number;
    stalled_packets: number;
    dependent_packets: number;
    duplicate_packets: number;
    crc_error_packets: number;
    parse_error_packets: number;
    invalid_neighbor_packets: number;
    last_packet_seq: number;
    last_rank_up_seq: number;
    progress: number;
    complete: boolean;
  };
  recovered_data(): Uint8Array | null | undefined;
  reset(): void;
};

type WasmEncoderCtor = new (sampleRate: number) => WasmEncoderLike;
type WasmDecoderCtor = new (sampleRate: number) => WasmDecoderLike;
type WasmInitFn = () => Promise<unknown>;
type WasmBindings = {
  init: WasmInitFn;
  WasmEncoder: WasmEncoderCtor;
  WasmDecoder: WasmDecoderCtor;
  flavor: "base" | "simd";
};

const SIMD_PROBE_WASM = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d,
  0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x00,
  0x0a, 0x19, 0x01, 0x17, 0x00,
  0xfd, 0x0c,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfd, 0x15, 0x00,
  0x0b,
]);

const supportsWasmSimd = () => {
  if (typeof WebAssembly === "undefined" || typeof WebAssembly.validate !== "function") {
    return false;
  }
  try {
    return WebAssembly.validate(SIMD_PROBE_WASM);
  } catch {
    return false;
  }
};

const BASE_BINDINGS: WasmBindings = {
  init: initBase as WasmInitFn,
  WasmEncoder: WasmEncoderBase as unknown as WasmEncoderCtor,
  WasmDecoder: WasmDecoderBase as unknown as WasmDecoderCtor,
  flavor: "base",
};

const SIMD_BINDINGS: WasmBindings = {
  init: initSimd as WasmInitFn,
  WasmEncoder: WasmEncoderSimd as unknown as WasmEncoderCtor,
  WasmDecoder: WasmDecoderSimd as unknown as WasmDecoderCtor,
  flavor: "simd",
};

type RecyclePortInboundMessage = 
  | { type: "recycle"; data: Float32Array }
  | { type: "input"; data: Float32Array };

export class MistcastBackend {
  private encoder: WasmEncoderLike | null = null;
  private decoder: WasmDecoderLike | null = null;
  private audioOutPort: MessagePort | null = null;
  private audioInPort: MessagePort | null = null;
  private audioPacketSender: RecycleTransferSender | null = null;
  private wasmInitialized = false;
  private wasmBindings: WasmBindings | null = null;
  private isEncoding = false;

  private onPacket: ((data: Uint8Array) => void) | null = null;
  private onProgress: ((p: any) => void) | null = null;

  private async ensureWasm() {
    if (!this.wasmBindings) {
      this.wasmBindings = supportsWasmSimd() ? SIMD_BINDINGS : BASE_BINDINGS;
    }
    if (this.wasmInitialized) return;

    if (this.wasmBindings.flavor === "simd") {
      try {
        await this.wasmBindings.init();
      } catch (e) {
        console.warn("[Worker] SIMD wasm init failed. Falling back to base.", e);
        this.wasmBindings = BASE_BINDINGS;
        await this.wasmBindings.init();
      }
    } else {
      await this.wasmBindings.init();
    }

    this.wasmInitialized = true;
    console.log(`[Worker] WASM Initialized (${this.wasmBindings.flavor})`);
  }

  private requireBindings(): WasmBindings {
    if (!this.wasmBindings) throw new Error("WASM is not initialized");
    return this.wasmBindings;
  }

  async init() {
    await this.ensureWasm();
  }

  async setAudioOutPort(port: MessagePort) {
    if (this.audioOutPort) {
      this.audioOutPort.onmessage = null;
      this.audioOutPort.close();
    }
    this.audioOutPort = port;
    this.audioOutPort.onmessage = (event: MessageEvent) => {
      const msg = event.data as RecyclePortInboundMessage | null;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "recycle" && msg.data instanceof Float32Array) {
        this.audioPacketSender?.recycle(msg.data);
        this.fillEncoderBuffer();
      }
    };
    this.audioOutPort.start();
  }

  async setAudioInPort(port: MessagePort) {
    if (this.audioInPort) {
      this.audioInPort.onmessage = null;
      this.audioInPort.close();
    }
    this.audioInPort = port;
    this.audioInPort.onmessage = (event: MessageEvent) => {
      const msg = event.data as RecyclePortInboundMessage | null;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "input" && msg.data instanceof Float32Array) {
        this.processSamples(msg.data);
      }
    };
    this.audioInPort.start();
  }

  async startEncoder(data: Uint8Array, sampleRate: number) {
    await this.init();
    const bindings = this.requireBindings();
    
    // 前回の送信状態をリセット
    this.isEncoding = false;
    this.audioOutPort?.postMessage({ type: "reset" });

    if (!this.encoder) {
        this.encoder = new bindings.WasmEncoder(sampleRate);
    }
    this.encoder.set_data(data);
    console.log(`[Worker] Encoder started (size=${data.length}, rate=${sampleRate})`);

    const dummyFrame = this.encoder.pull_frame();
    if (!dummyFrame) return;
    
    if (!this.audioPacketSender || (this.audioPacketSender as any).packetSamples !== dummyFrame.length) {
        this.audioPacketSender = new RecycleTransferSender(dummyFrame.length, 64);
    }

    this.isEncoding = true;
    this.fillEncoderBuffer();
  }

  private fillEncoderBuffer() {
    if (!this.isEncoding || !this.encoder || !this.audioPacketSender || !this.audioOutPort) return;

    while (this.isEncoding) {
        const frame = this.encoder.pull_frame();
        if (!frame) break;

        const samples = new Float32Array(frame);
        const prevDropped = this.audioPacketSender.getDroppedSamplesCount();
        this.audioPacketSender.appendFrom(samples, samples.length, 1, this.audioOutPort);

        if (this.audioPacketSender.getDroppedSamplesCount() > prevDropped) {
            break; 
        }
    }
  }

  async stopEncoder() {
    this.isEncoding = false;
    this.audioOutPort?.postMessage({ type: "reset" });
  }

  async startDecoder(sampleRate: number, 
                     onPacket: (data: Uint8Array) => void,
                     onProgress: (p: any) => void) {
    await this.init();
    const bindings = this.requireBindings();
    console.log(`[Worker] Decoder setup (fixed protocol, rate=${sampleRate})`);
    
    this.decoder = new bindings.WasmDecoder(sampleRate);
    this.onPacket = onPacket;
    this.onProgress = onProgress;
  }

  async processSamples(samples: Float32Array) {
    if (!this.decoder) return null;
    
    const progress = this.decoder.process_samples(samples);
    const result = {
        received: progress.received_packets,
        needed: progress.needed_packets,
        rank: progress.rank_packets,
        stalled: progress.stalled_packets,
        dependent: progress.dependent_packets,
        duplicate: progress.duplicate_packets,
        crcErrors: progress.crc_error_packets,
        parseErrors: progress.parse_error_packets,
        invalidNeighbors: progress.invalid_neighbor_packets,
        lastPacketSeq: progress.last_packet_seq,
        lastRankUpSeq: progress.last_rank_up_seq,
        progress: progress.progress,
        complete: progress.complete,
    };

    if (this.onProgress) {
        this.onProgress(result);
    }

    if (progress.complete) {
      console.log("[Worker] Decode complete!");
      const data = this.decoder.recovered_data();
      if (data && this.onPacket) {
        this.onPacket(data);
      }
      this.decoder = null; 
    }
    
    return result;
  }

  async resetDecoder() {
    if (this.decoder) {
        this.decoder.reset();
        this.decoder = null;
    }
  }
}

expose(new MistcastBackend());
