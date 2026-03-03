<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import { useDemoRuntime } from '../demo-runtime';
import { extractImagePayload, type ImagePayload } from '../utils/image-detector';
import SpectrumCanvas from './SpectrumCanvas.vue';

const runtime = useDemoRuntime();

type InputMode = 'loopback' | 'mic';

const outputText = ref('');
const outputImageUrl = ref('');
const outputImageMime = ref('');
const outputBinaryData = ref<Uint8Array | null>(null);
const outputBinarySize = ref(0);

const receivedPackets = ref(0);
const totalNeededPackets = ref(0);
const rankPackets = ref(0);
const stalledPackets = ref(0);
const dependentPackets = ref(0);
const duplicatePackets = ref(0);
const crcErrorPackets = ref(0);
const parseErrorPackets = ref(0);
const invalidNeighborPackets = ref(0);
const lastPacketSeq = ref(-1);
const lastRankUpSeq = ref(-1);
const progressPercent = ref(0);

const basisMatrixK = ref(0);

const decoderProcAvgMs = ref(0);
const decoderProcMaxMs = ref(0);
const decoderProcLastMs = ref(0);
const decoderProcBlockMs = ref(0);
const decoderProcOverruns = ref(0);
const decoderProcInputRms = ref(0);
const decoderProcBlocks = ref(0);

const rxLogs = ref<string[]>([]);
const rxTick = ref(0);
const rxNoChangeTicks = ref(0);
const rxLogCopied = ref(false);
let rxLogCopiedTimer: number | null = null;

// 重複確率を計算: 1/256^(k-r) * 100
const stallProbability = computed(() => {
  const k = totalNeededPackets.value;
  const r = rankPackets.value;
  if (k === 0) return '0.0000';
  if (k <= r) return '100.00';
  const prob = (1 / Math.pow(256, k - r)) * 100;
  // 0.0001% 以上なら固定小数点、未満なら指数表示
  if (prob >= 0.0001) {
    return prob.toFixed(4);
  }
  const exp = prob.toExponential(2);
  const match = exp.match(/(\d)\d*\.\d*e([+-]\d+)$/);
  if (match) {
    const n = match[1];
    const exponent = parseInt(match[2], 10);
    return `${n}e${exponent}`;
  }
  return exp;
});

const receiverStatus = ref('Idle');
const isTogglingMic = ref(false);

const inputMode = ref<InputMode>('loopback');
const isMicActive = computed(() => inputMode.value === 'mic');

const basisCanvas = ref<HTMLCanvasElement | null>(null);
const rxSpectrumCanvas = ref<HTMLCanvasElement | null>(null);

let receiverWorker: Worker | null = null;
let receiverBackend: Comlink.Remote<MistcastBackend> | null = null;
let decoderNode: AudioWorkletNode | null = null;
let decoderStreamSink: MediaStreamAudioDestinationNode | null = null;
let rxAnalyserNode: AnalyserNode | null = null;
let rxFftData: Float32Array | null = null;
let rxFftRafId: number | null = null;
let rxInputGain: GainNode | null = null;
let micSource: MediaStreamAudioSourceNode | null = null;
let micStream: MediaStream | null = null;
let demoAirGapNodeRef: GainNode | null = null;

let opQueue: Promise<void> = Promise.resolve();

function clearOutput() {
  outputText.value = '';
  outputImageMime.value = '';
  outputBinaryData.value = null;
  outputBinarySize.value = 0;
  if (outputImageUrl.value) {
    URL.revokeObjectURL(outputImageUrl.value);
    outputImageUrl.value = '';
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const next = opQueue.then(() => fn(), () => fn());
  opQueue = next.then(
    () => undefined,
    () => undefined
  );
  return next;
}

function safeDisconnect<T extends AudioNode>(node: T | null, destination?: AudioNode | null) {
  if (!node) return;
  try {
    if (destination) node.disconnect(destination);
    else node.disconnect();
  } catch {
    // no-op
  }
}

function downloadBinary() {
  if (!outputBinaryData.value) return;
  const blob = new Blob([outputBinaryData.value as BlobPart], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'received.bin';
  a.click();
  URL.revokeObjectURL(url);
}

function setDecodedOutput(recovered: Uint8Array) {
  clearOutput();

  if (recovered.length === 0) return;

  // 画像を先にチェック
  const image = extractImagePayload(recovered);
  if (image) {
    outputImageMime.value = image.mime;
    outputImageUrl.value = URL.createObjectURL(
      new Blob([image.bytes as BlobPart], { type: image.mime })
    );
    return;
  }

  // テキストとして処理
  try {
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const text = decoder.decode(recovered);
    // null文字が含まれておらず、制御文字が少なければテキストとみなす
    if (!text.includes('\0')) {
      outputText.value = text;
      return;
    }
  } catch {
    // UTF-8として不正な場合はバイナリ扱い
  }

  outputBinaryData.value = recovered;
  outputBinarySize.value = recovered.length;
}

function pushRxLog(line: string) {
  rxLogs.value.push(line);
  if (rxLogs.value.length > 120) {
    rxLogs.value.splice(0, rxLogs.value.length - 120);
  }
}

async function copyRxLogs() {
  const text = rxLogs.value.join('\n');
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
  rxLogCopied.value = true;
  if (rxLogCopiedTimer !== null) {
    window.clearTimeout(rxLogCopiedTimer);
  }
  rxLogCopiedTimer = window.setTimeout(() => {
    rxLogCopied.value = false;
    rxLogCopiedTimer = null;
  }, 1200);
}

function resetDecoderProgressState(clearLogs: boolean) {
  receivedPackets.value = 0;
  totalNeededPackets.value = 0;
  rankPackets.value = 0;
  stalledPackets.value = 0;
  dependentPackets.value = 0;
  duplicatePackets.value = 0;
  crcErrorPackets.value = 0;
  parseErrorPackets.value = 0;
  invalidNeighborPackets.value = 0;
  lastPacketSeq.value = -1;
  lastRankUpSeq.value = -1;
  progressPercent.value = 0;
  if (clearLogs) {
    rxLogs.value = [];
    rxTick.value = 0;
    rxNoChangeTicks.value = 0;
  }
}

function resetDecoderProcessorStats() {
  decoderProcAvgMs.value = 0;
  decoderProcMaxMs.value = 0;
  decoderProcLastMs.value = 0;
  decoderProcBlockMs.value = 0;
  decoderProcOverruns.value = 0;
  decoderProcInputRms.value = 0;
  decoderProcBlocks.value = 0;
}

function makeOnPacketCallback() {
  return Comlink.proxy((recovered: Uint8Array) => {
    setDecodedOutput(recovered);
    receiverStatus.value = 'Decoded!';
  });
}

function makeOnProgressCallback() {
  return Comlink.proxy((p: any) => {
    const prevReceived = receivedPackets.value;
    const prevRank = rankPackets.value;

    receivedPackets.value = p.received;
    totalNeededPackets.value = p.needed;
    rankPackets.value = p.rank ?? 0;
    stalledPackets.value = p.stalled ?? Math.max(0, receivedPackets.value - rankPackets.value);
    dependentPackets.value = p.dependent ?? stalledPackets.value;
    duplicatePackets.value = p.duplicate ?? 0;
    crcErrorPackets.value = p.crcErrors ?? 0;
    parseErrorPackets.value = p.parseErrors ?? 0;
    invalidNeighborPackets.value = p.invalidNeighbors ?? 0;
    lastPacketSeq.value = p.lastPacketSeq ?? -1;
    lastRankUpSeq.value = p.lastRankUpSeq ?? -1;
    progressPercent.value = p.progress;

    if (p.basisMatrix && basisCanvas.value) {
      const matrix = p.basisMatrix as Uint8Array;
      const k = Math.sqrt(matrix.length);
      basisMatrixK.value = Math.floor(k);
      const canvas = basisCanvas.value;
      const ctx = canvas.getContext('2d');
      if (ctx && k > 0) {
        const dpr = window.devicePixelRatio || 1;
        const scale = 4; // 1係数あたりの論理ピクセル数
        const logicalSize = k * scale;
        const physicalSize = Math.floor(logicalSize * dpr);

        if (canvas.width !== physicalSize || canvas.height !== physicalSize) {
          canvas.width = physicalSize;
          canvas.height = physicalSize;
        }

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, logicalSize, logicalSize);

        // 対角線の左側（下三角）を薄い灰色で塗る
        ctx.fillStyle = '#f1f5f9';
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, logicalSize);
        ctx.lineTo(logicalSize, logicalSize);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#0f6bd7';
        for (let row = 0; row < k; row++) {
          const rowOffset = row * k;
          for (let col = 0; col < k; col++) {
            if (matrix[rowOffset + col] !== 0) {
              ctx.fillRect(col * scale, row * scale, scale, scale);
            }
          }
        }
      }
    }

    const proc = p.decoderProc;
    if (proc) {
      decoderProcAvgMs.value = proc.avgProcessMs ?? 0;
      decoderProcMaxMs.value = proc.maxProcessMs ?? 0;
      decoderProcLastMs.value = proc.lastProcessMs ?? 0;
      decoderProcBlockMs.value = proc.blockDurationMs ?? 0;
      decoderProcOverruns.value = proc.overruns ?? 0;
      decoderProcInputRms.value = proc.inputRms ?? 0;
      decoderProcBlocks.value = proc.blocks ?? 0;
    }

    rxTick.value += 1;
    const changed =
      prevReceived !== receivedPackets.value ||
      prevRank !== rankPackets.value ||
      p.complete;
    if (changed) {
      rxNoChangeTicks.value = 0;
      pushRxLog(
        `#${rxTick.value} recv=${receivedPackets.value} rank=${rankPackets.value}/${totalNeededPackets.value} stall=${stalledPackets.value} dup=${duplicatePackets.value} crc=${crcErrorPackets.value} parse=${parseErrorPackets.value} invN=${invalidNeighborPackets.value} prog=${(progressPercent.value * 100).toFixed(1)}% lastSeq=${lastPacketSeq.value} lastRankUp=${lastRankUpSeq.value}${p.complete ? ' COMPLETE' : ''}`
      );
      return;
    }

    rxNoChangeTicks.value += 1;
    if (rxNoChangeTicks.value % 64 === 0) {
      pushRxLog(
        `#${rxTick.value} heartbeat no-progress=${rxNoChangeTicks.value} recv=${receivedPackets.value} rank=${rankPackets.value}/${totalNeededPackets.value} rms=${decoderProcInputRms.value.toFixed(5)} avgMs=${decoderProcAvgMs.value.toFixed(3)} overrun=${decoderProcOverruns.value}`
      );
    }
  });
}

async function teardownReceiverGraph() {
  if (receiverBackend) {
    try {
      await receiverBackend.resetDecoder();
    } catch {
      // no-op
    }
  }

  if (receiverWorker) {
    receiverWorker.terminate();
    receiverWorker = null;
  }
  receiverBackend = null;

  safeDisconnect(demoAirGapNodeRef, rxInputGain);
  safeDisconnect(micSource, rxInputGain);
  safeDisconnect(micSource);
  safeDisconnect(rxInputGain, rxAnalyserNode);
  safeDisconnect(rxInputGain, decoderNode);
  safeDisconnect(rxInputGain);
  safeDisconnect(decoderNode, decoderStreamSink);
  safeDisconnect(decoderNode);
  safeDisconnect(rxAnalyserNode);

  micSource = null;
  rxInputGain = null;
  decoderNode = null;
  rxAnalyserNode = null;
  decoderStreamSink = null;
  demoAirGapNodeRef = null;
}

async function rebuildReceiverGraph() {
  const { audioContext, demoAirGapNode } = await runtime.ensureAudioCore();

  await teardownReceiverGraph();

  receiverWorker = new MistcastWorker();
  receiverBackend = Comlink.wrap<MistcastBackend>(receiverWorker);
  await receiverBackend.init();

  decoderNode = new AudioWorkletNode(audioContext, 'decoder-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
    channelCountMode: 'explicit',
    channelInterpretation: 'discrete',
  });
  await receiverBackend.setAudioInPort(Comlink.transfer(decoderNode.port, [decoderNode.port]));

  rxInputGain = audioContext.createGain();
  rxInputGain.gain.value = 1.0;

  rxAnalyserNode = audioContext.createAnalyser();
  rxAnalyserNode.fftSize = 4096;
  rxAnalyserNode.smoothingTimeConstant = 0.6;
  rxAnalyserNode.minDecibels = -100;
  rxAnalyserNode.maxDecibels = -20;

  rxInputGain.connect(rxAnalyserNode);
  rxInputGain.connect(decoderNode);

  decoderStreamSink = audioContext.createMediaStreamDestination();
  decoderNode.connect(decoderStreamSink);

  demoAirGapNodeRef = demoAirGapNode;

  if (inputMode.value === 'mic') {
    if (!micStream) {
      throw new Error('mic stream is not available');
    }
    micSource = audioContext.createMediaStreamSource(micStream);
    micSource.connect(rxInputGain);
    receiverStatus.value = 'Mic Active (Rx)';
  } else {
    demoAirGapNode.connect(rxInputGain);
    receiverStatus.value = 'Internal Loopback';
  }

  await receiverBackend.startDecoder(
    audioContext.sampleRate,
    makeOnPacketCallback(),
    makeOnProgressCallback(),
    runtime.modemMode.value
  );
}

async function toggleMic() {
  await runExclusive(async () => {
    if (!runtime.coreReady.value) return;
    isTogglingMic.value = true;
    try {
      const { audioContext } = await runtime.ensureAudioCore();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      if (inputMode.value === 'loopback') {
        try {
          micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: 48000,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
          inputMode.value = 'mic';
        } catch (e) {
          console.error(e);
          receiverStatus.value = 'Mic Error';
          return;
        }
      } else {
        micStream?.getTracks().forEach((t) => t.stop());
        micStream = null;
        inputMode.value = 'loopback';
      }

      await rebuildReceiverGraph();
    } finally {
      isTogglingMic.value = false;
    }
  });
}

async function reset() {
  await runExclusive(async () => {
    decoderNode?.port.postMessage({ type: 'reset' });
    await receiverBackend?.resetDecoder();
    clearOutput();
    resetDecoderProgressState(true);
    resetDecoderProcessorStats();
    await rebuildReceiverGraph();
    receiverStatus.value = inputMode.value === 'mic' ? 'Mic Active (Rx)' : 'Ready (Rx standby)';
  });
}

watch(
  () => runtime.coreReady.value,
  async (ready) => {
    if (!ready) return;
    await runExclusive(async () => {
      if (!receiverBackend) {
        await rebuildReceiverGraph();
        receiverStatus.value = inputMode.value === 'mic' ? 'Mic Active (Rx)' : 'Ready (Rx standby)';
      }
    });
  },
  { immediate: true }
);

watch(
  () => runtime.modemMode.value,
  async () => {
    if (!runtime.coreReady.value) return;
    await reset();
  }
);

watch(
  [receivedPackets, progressPercent],
  ([received, progress]) => {
    // パケットを1つでも受信しており、かつ完了していない場合は「ビジー」とみなす
    const active = received > 0 && progress < 1.0;
    // 送信側がビジーでない場合のみ、受信側の状態で上書きする（簡易的な論理和）
    if (!active && runtime.isBusy.value && receivedPackets.value > 0) {
       runtime.isBusy.value = false;
    } else if (active) {
       runtime.isBusy.value = true;
    }
  }
);

onBeforeUnmount(() => {
  if (rxLogCopiedTimer !== null) {
    window.clearTimeout(rxLogCopiedTimer);
  }
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
  void teardownReceiverGraph();
  clearOutput();
});
</script>

<template>
  <section class="panel receiver-panel">
    <div class="receiver-header">
      <div>
        <div class="receiver-title-row">
          <h2>Receiver</h2>
          <div class="status-chip" :class="receiverStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
            {{ receiverStatus }}
          </div>
        </div>
        <div class="path-banner">
          <span class="path-label">Input Path</span>
          <code v-if="!isMicActive">[demoAirGapNode] -digital- [Receiver]</code>
          <code v-else>[Mic] -acoustic- [Receiver]</code>
        </div>
      </div>
      <div class="button-row compact button-row-2col">
        <button @click="toggleMic" :class="{ 'btn-active': isMicActive }" class="btn" :disabled="!runtime.coreReady.value || isTogglingMic">
          {{ isTogglingMic ? 'Switching...' : (isMicActive ? 'Disable Mic' : 'Enable Mic') }}
        </button>
        <button @click="reset" class="btn" :disabled="!runtime.coreReady.value || isTogglingMic">Clear</button>
      </div>
    </div>

    <div class="progress-block">
      <div class="progress-head">
        <span>Rank {{ rankPackets }} / {{ totalNeededPackets || '?' }}</span>
        <span>{{ (progressPercent * 100).toFixed(1) }}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" :style="{ width: `${progressPercent * 100}%` }" />
      </div>
      <div class="basis-panel" v-if="progressPercent < 1.0 && receivedPackets > 0" data-tooltip="ガウスの消去法によるランク更新の可視化。各パケットは行ベクトルとして表現され、青いセルは非ゼロ要素を表します。左下三角領域（灰色）は、前進消去によって常にゼロに保たれる領域です。白色のセルはまだ処理されていない上三角部分のゼロ要素です。">
        <p class="basis-title">Basis Matrix (Gaussian Elimination) ({{ basisMatrixK }}x{{ basisMatrixK }}, {{ basisMatrixK * 16 * 8 }}bits)</p>
        <canvas ref="basisCanvas" class="basis-canvas"></canvas>
      </div>
      <div class="display" v-if="progressPercent >= 1.0">
        <p class="display-title">Decoded Result</p>
        <pre v-if="outputText">{{ outputText }}</pre>
        <div v-else-if="outputImageUrl" class="image-result">
          <img :src="outputImageUrl" :alt="`decoded image (${outputImageMime || 'unknown'})`" />
          <p class="image-meta">{{ outputImageMime }}</p>
        </div>
        <div v-else-if="outputBinaryData" class="binary-result">
          <button @click="downloadBinary" class="btn btn-primary">
            [received.bin: {{ formatSize(outputBinarySize) }}]
          </button>
        </div>
        <p v-else class="placeholder">No decoded data</p>
      </div>
    </div>

    <SpectrumCanvas
      :analyser-node="rxAnalyserNode"
      title="Receiver FFT (Linear Frequency Axis)"
    />

    <div class="metric-grid">
      <div class="metric" data-tooltip="正常に受信してシステムに受け入れられたパケットの総数"><span>Accepted</span><strong>{{ receivedPackets }}</strong></div>
      <div class="metric" data-tooltip="受信したが、既存のパケットと線形従属の関係にあるため行列のランク上昇に寄与していないパケット数。GF(256) で必要なパケット数が k のとき、現在のランクが r だと次のパケットが従属になる確率は約 1/256^(k-r)"><span>Stall</span><strong>{{ stalledPackets }}</strong><small>(現在重複確率: {{ stallProbability }}%)</small></div>
      <div class="metric" data-tooltip="他のパケットと線形従属の関係にあり、行列のランクを上げるためにまだ他のパケットの受信を待っているパケット数"><span>Dep</span><strong>{{ dependentPackets }}</strong></div>
      <div class="metric" data-tooltip="以前に受信したパケットと同じシーケンス番号を持つ重複パケット数。シーケンス番号は16bit（0-65535）で、65536パケット送信すると一周して重複が発生する可能性あり"><span>Dup</span><strong>{{ duplicatePackets }}</strong></div>
      <div class="metric" data-tooltip="CRCチェックに失敗したパケット数（ノイズや伝送エラーでデータが破損）"><span>CRC</span><strong>{{ crcErrorPackets }}</strong></div>
      <div class="metric" data-tooltip="パケットの解析に失敗したパケット数（フォーマット不正や構造エラー）"><span>Parse</span><strong>{{ parseErrorPackets }}</strong></div>
      <div class="metric" data-tooltip="近隣パケット間の整合性チェックに失敗したパケット数"><span>InvNbr</span><strong>{{ invalidNeighborPackets }}</strong></div>
      <div class="metric" data-tooltip="最後に受信したパケットのシーケンス番号（-1は未受信）"><span>Last Seq</span><strong>{{ lastPacketSeq }}</strong></div>
      <div class="metric" data-tooltip="最後に行列のランクが上昇した際のパケットシーケンス番号（復号の進捗指標）"><span>Last RankUp</span><strong>{{ lastRankUpSeq }}</strong></div>
    </div>

    <div class="proc-stats">
      <p class="proc-title">DecoderProcessor Timing</p>
      <div class="proc-grid">
        <div data-tooltip="1ブロックあたりの平均処理時間（全ブロックの平均値）"><span>avg</span><strong>{{ decoderProcAvgMs.toFixed(3) }} ms</strong></div>
        <div data-tooltip="1ブロックの処理時間の最大値（ピーク負荷時のパフォーマンス指標）"><span>max</span><strong>{{ decoderProcMaxMs.toFixed(3) }} ms</strong></div>
        <div data-tooltip="直近のブロック処理時間（最新の処理パフォーマンス）"><span>last</span><strong>{{ decoderProcLastMs.toFixed(3) }} ms</strong></div>
        <div data-tooltip="1ブロックの処理に割り当てられた時間バジェット（サンプリングレートとバッファサイズから算出）"><span>budget</span><strong>{{ decoderProcBlockMs.toFixed(3) }} ms</strong></div>
        <div data-tooltip="処理時間がバジェットを超過した回数（リアルタイム処理失敗の指標、超過するとパケット損失の原因）"><span>overrun</span><strong>{{ decoderProcOverruns }}</strong></div>
        <div data-tooltip="入力信号の実効値（0に近いほど静寂、大きいほど信号あり）"><span>input RMS</span><strong>{{ decoderProcInputRms.toFixed(5) }}</strong></div>
        <div data-tooltip="処理したオーディオブロックの総数"><span>blocks</span><strong>{{ decoderProcBlocks }}</strong></div>
      </div>
    </div>

    <div class="rx-log" v-if="rxLogs.length > 0">
      <div class="rx-log-header">
        <span>Rx Log</span>
        <button @click="copyRxLogs" class="btn btn-xs">{{ rxLogCopied ? 'Copied' : 'Copy' }}</button>
      </div>
      <pre>{{ rxLogs.join('\n') }}</pre>
    </div>
  </section>
</template>
