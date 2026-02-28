/// <reference lib="webworker" />
import {
  RecycleTransferReceiver,
  type RecycleTransferInputMessage,
  type RecycleTransferRecycleMessage,
} from "./recycle-transfer-bridge";

declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: unknown);
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

const workletGlobal = globalThis as unknown as { sampleRate: number; currentTime: number };

/**
 * EncoderProcessor: Workerから届いた「送信波形」を再生するだけのノード
 */
export class EncoderProcessor extends AudioWorkletProcessor {
  private readonly packetQueue = new RecycleTransferReceiver();
  private inputPort: MessagePort | null = null;
  private started = false;
  private lastSample = 0.0;
  private readonly sampleRateHz = workletGlobal.sampleRate;

  // バッファ管理（refs/radio 基準）
  private baseMinStartSamples = Math.floor(this.sampleRateHz * 0.1);
  private lowWaterSamples = Math.floor(this.sampleRateHz * 0.03);

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      if (!msg) return;
      if (msg.type === "attach-input-port" && msg.port) {
        this.inputPort = msg.port;
        if (this.inputPort) {
          this.inputPort.onmessage = (e: MessageEvent) => {
            const innerMsg = e.data as RecycleTransferInputMessage;
            if (innerMsg.type === "push") {
              this.packetQueue.pushFromMessage(innerMsg);
            } else if (innerMsg.type === "reset") {
              this.packetQueue.reset((data) => this.recycleChunkData(data));
              this.started = false;
            }
          };
          this.inputPort.start();
        }
      }
    };
  }

  private recycleChunkData(data: Float32Array): void {
    if (!this.inputPort) return;
    const msg: RecycleTransferRecycleMessage = { type: "recycle", data };
    this.inputPort.postMessage(msg, [data.buffer]);
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const outputBus = outputs[0];
    const outL = outputBus?.[0];
    if (!outL) return true;
    const outR = outputBus?.[1] ?? outL;

    outL.fill(0);
    if (outR !== outL) outR.fill(0);

    if (!this.started) {
      if (this.packetQueue.getBufferedFrames() < this.baseMinStartSamples) {
        return true;
      }
      this.started = true;
    }

    const drained = this.packetQueue.drainInto(outL, outR, (data) => this.recycleChunkData(data));
    if (drained.writtenFrames > 0) {
      this.lastSample = drained.lastSample;
    }

    if (drained.writtenFrames < outL.length) {
      // 供給不足（Underrun）時は最後のサンプルで補間
      outL.fill(this.lastSample, drained.writtenFrames);
      outR.fill(this.lastSample, drained.writtenFrames);
      if (this.packetQueue.getBufferedFrames() < this.lowWaterSamples) {
        this.started = false;
      }
    }

    return true;
  }
}

/**
 * DecoderProcessor: マイク等の入力を一定量蓄積して Worker へ送るだけのノード
 */
export class DecoderProcessor extends AudioWorkletProcessor {
  private outputPort: MessagePort | null = null;
  private readonly CHUNK_SIZE = 4096; // 同期捕捉に十分なサイズ
  private buffer = new Float32Array(this.CHUNK_SIZE);
  private pos = 0;

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "attach-output-port" && msg.port) {
        this.outputPort = msg.port;
        if (this.outputPort) {
          this.outputPort.start();
        }
      }
    };
  }

  process(inputs: Float32Array[][]): boolean {
    const inputBus = inputs[0];
    const inL = inputBus?.[0];
    if (!inL || !this.outputPort) return true;

    for (let i = 0; i < inL.length; i++) {
      this.buffer[this.pos++] = inL[i];
      if (this.pos >= this.CHUNK_SIZE) {
        // 蓄積完了、Workerへ転送
        // 所有権転送 (Transferable) を使用してアロケーションを抑える
        const transferBuffer = new Float32Array(this.buffer);
        this.outputPort.postMessage({ type: "input", data: transferBuffer }, [transferBuffer.buffer]);
        this.pos = 0;
      }
    }

    return true;
  }
}

const maybeRegister = (
  globalThis as unknown as {
    registerProcessor?: (
      name: string,
      ctor: new () => AudioWorkletProcessor,
    ) => void;
  }
).registerProcessor;

if (typeof maybeRegister === "function") {
  maybeRegister("encoder-processor", EncoderProcessor);
  maybeRegister("decoder-processor", DecoderProcessor);
}
