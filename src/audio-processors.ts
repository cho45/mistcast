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

const nowMs = (): number => {
  const perf = (globalThis as unknown as { performance?: { now?: () => number } }).performance;
  if (perf?.now) return perf.now();
  return Date.now();
};

/**
 * EncoderProcessor: Workerから届いた「送信波形」を再生するノード
 * 自身の node.port を通じてメッセージを受け取る
 */
export class EncoderProcessor extends AudioWorkletProcessor {
  private readonly packetQueue = new RecycleTransferReceiver();
  private started = false;
  private lastSample = 0.0;
  private readonly sampleRateHz = workletGlobal.sampleRate;

  private baseMinStartSamples = Math.floor(this.sampleRateHz * 0.1);
  private lowWaterSamples = Math.floor(this.sampleRateHz * 0.03);

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent) => {
      const msg = e.data as RecycleTransferInputMessage;
      if (msg.type === "push") {
        this.packetQueue.pushFromMessage(msg);
      } else if (msg.type === "reset") {
        this.packetQueue.reset((data) => this.recycleChunkData(data));
        this.started = false;
        this.lastSample = 0.0;
      }
    };
  }

  private recycleChunkData(data: Float32Array): void {
    const msg: RecycleTransferRecycleMessage = { type: "recycle", data };
    this.port.postMessage(msg, [data.buffer]);
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
 * DecoderProcessor: 入力を蓄積して 自身の port へ送るノード
 */
export class DecoderProcessor extends AudioWorkletProcessor {
  private readonly CHUNK_SIZE = 4096;
  private readonly REPORT_EVERY_BLOCKS = 64;
  private buffer = new Float32Array(this.CHUNK_SIZE);
  private pos = 0;
  private statsBlocks = 0;
  private statsTotalMs = 0;
  private statsMaxMs = 0;
  private statsOverruns = 0;
  private statsLastMs = 0;
  private statsInputRms = 0;

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent) => {
      const msg = e.data as { type?: string } | null;
      if (msg?.type === "reset") {
        this.pos = 0;
        this.buffer.fill(0);
        this.statsBlocks = 0;
        this.statsTotalMs = 0;
        this.statsMaxMs = 0;
        this.statsOverruns = 0;
        this.statsLastMs = 0;
        this.statsInputRms = 0;
      }
    };
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const t0 = nowMs();
    const inputBus = inputs[0];
    const inL = inputBus?.[0];

    const outputBus = outputs[0];
    const outL = outputBus?.[0];
    const outR = outputBus?.[1] ?? outL;
    if (outL) {
      outL.fill(0);
      if (outR && outR !== outL) outR.fill(0);
    }

    if (!inL) return true;

    let rmsAcc = 0;
    for (let i = 0; i < inL.length; i++) {
      const v = inL[i];
      this.buffer[this.pos++] = v;
      rmsAcc += v * v;
      if (this.pos >= this.CHUNK_SIZE) {
        const transferBuffer = new Float32Array(this.buffer);
        // 標準の port を使用
        this.port.postMessage({ type: "input", data: transferBuffer }, [transferBuffer.buffer]);
        this.pos = 0;
      }
    }

    const elapsedMs = nowMs() - t0;
    const blockMs = (inL.length / workletGlobal.sampleRate) * 1000;
    const inputRms = Math.sqrt(rmsAcc / inL.length);
    this.statsBlocks += 1;
    this.statsTotalMs += elapsedMs;
    this.statsMaxMs = Math.max(this.statsMaxMs, elapsedMs);
    this.statsLastMs = elapsedMs;
    this.statsInputRms = inputRms;
    if (elapsedMs > blockMs) {
      this.statsOverruns += 1;
    }

    if (this.statsBlocks % this.REPORT_EVERY_BLOCKS === 0) {
      this.port.postMessage({
        type: "stats",
        blocks: this.statsBlocks,
        avgProcessMs: this.statsTotalMs / this.statsBlocks,
        maxProcessMs: this.statsMaxMs,
        lastProcessMs: this.statsLastMs,
        blockDurationMs: blockMs,
        overruns: this.statsOverruns,
        inputRms: this.statsInputRms,
      });
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
