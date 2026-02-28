import { expose } from "comlink";
import initWasm, { WasmEncoder, WasmDecoder, WasmDecodeProgress } from "../pkg/dsp";
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
  private onProgress: ((p: { received: number, needed: number, progress: f32, complete: boolean }) => void) | null = null;

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
    if (typeof this.audioOutPort.start === "function") {
      this.audioOutPort.start();
    }
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
    if (typeof this.audioInPort.start === "function") {
      this.audioInPort.start();
    }
  }

  async startEncoder(data: Uint8Array, sampleRate: number) {
    await this.init();
    if (!this.encoder) {
        this.encoder = new WasmEncoder(sampleRate);
    }
    this.encoder.set_data(data);
    console.log("[Worker] Encoder started with data length:", data.length);

    const dummyFrame = this.encoder.pull_frame();
    if (!dummyFrame) return;
    
    this.audioPacketSender = new RecycleTransferSender(dummyFrame.length, 64);
    this.audioPacketSender.recycle(new Float32Array(dummyFrame));

    this.isEncoding = true;
    this.fillEncoderBuffer();
  }

  private fillEncoderBuffer() {
    if (!this.isEncoding || !this.encoder || !this.audioPacketSender || !this.audioOutPort) return;

    while (true) {
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
    console.log("[Worker] Encoder stopped");
  }

  async startDecoder(dataSize: number, ltK: number, sampleRate: number, 
                     onPacket: (data: Uint8Array) => void,
                     onProgress: (p: any) => void) {
    await this.init();
    console.log(`[Worker] Decoder started (size=${dataSize}, k=${ltK})`);
    this.decoder = new WasmDecoder(dataSize, ltK, sampleRate);
    this.onPacket = onPacket;
    this.onProgress = onProgress;
  }

  async processSamples(samples: Float32Array) {
    if (!this.decoder) return;
    
    const progress = this.decoder.process_samples(samples);
    
    // UIに進捗を通知
    if (this.onProgress) {
        this.onProgress({
            received: progress.received_packets,
            needed: progress.needed_packets,
            progress: progress.progress,
            complete: progress.complete,
        });
    }

    if (progress.complete) {
      console.log("[Worker] Decode complete!");
      const data = this.decoder.recovered_data();
      if (data && this.onPacket) {
        this.onPacket(data);
      }
    }
  }

  async resetDecoder() {
    this.decoder?.reset();
    console.log("[Worker] Decoder reset");
  }
}

expose(new MistcastBackend());
