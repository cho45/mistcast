import { expose } from "comlink";
import initWasm, { WasmEncoder, WasmDecoder } from "../pkg/dsp";
import { RecycleTransferSender } from "./recycle-transfer-bridge";

type RecyclePortInboundMessage = 
  | { type: "recycle"; data: Float32Array }
  | { type: "input"; data: Float32Array };

export class MistcastBackend {
  private encoder: WasmEncoder | null = null;
  private decoder: WasmDecoder | null = null;
  private audioOutPort: MessagePort | null = null;
  private audioInPort: MessagePort | null = null;
  private audioPacketSender: RecycleTransferSender | null = null;
  private wasmInitialized = false;
  private isEncoding = false;

  private onPacket: ((data: Uint8Array) => void) | null = null;
  private onProgress: ((p: any) => void) | null = null;

  async init() {
    if (!this.wasmInitialized) {
      await initWasm();
      this.wasmInitialized = true;
      console.log("[Worker] WASM Initialized");
    }
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
    
    // 前回の送信状態をリセット
    this.isEncoding = false;
    this.audioOutPort?.postMessage({ type: "reset" });

    if (!this.encoder) {
        this.encoder = new WasmEncoder(sampleRate);
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
    console.log(`[Worker] Decoder setup (fixed protocol, rate=${sampleRate})`);
    
    this.decoder = new WasmDecoder(sampleRate);
    this.onPacket = onPacket;
    this.onProgress = onProgress;
  }

  async processSamples(samples: Float32Array) {
    if (!this.decoder) return null;
    
    const progress = this.decoder.process_samples(samples);
    const result = {
        received: progress.received_packets,
        needed: progress.needed_packets,
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
