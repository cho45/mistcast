<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import { useDemoRuntime } from '../demo-runtime';

const runtime = useDemoRuntime();

type InputMode = 'loopback' | 'mic';

type ImagePayload = {
  mime: string;
  bytes: Uint8Array;
};

const outputText = ref('');
const outputImageUrl = ref('');
const outputImageMime = ref('');

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
const receiverStatus = ref('Idle');

const inputMode = ref<InputMode>('loopback');
const isMicActive = computed(() => inputMode.value === 'mic');

let rxLogCopiedTimer: number | null = null;
let receiverWorker: Worker | null = null;
let receiverBackend: Comlink.Remote<MistcastBackend> | null = null;
let decoderNode: AudioWorkletNode | null = null;
let rxInputGain: GainNode | null = null;
let decoderStreamSink: MediaStreamAudioDestinationNode | null = null;
let demoAirGapNodeRef: GainNode | null = null;
let micSource: MediaStreamAudioSourceNode | null = null;
let micStream: MediaStream | null = null;

let opQueue: Promise<void> = Promise.resolve();

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

function clearOutput() {
  outputText.value = '';
  outputImageMime.value = '';
  if (outputImageUrl.value) {
    URL.revokeObjectURL(outputImageUrl.value);
    outputImageUrl.value = '';
  }
}

function trimTrailingZeros(data: Uint8Array): Uint8Array {
  let last = data.length;
  while (last > 0 && data[last - 1] === 0) last--;
  return data.slice(0, last);
}

function detectImageMime(data: Uint8Array): string | null {
  if (
    data.length >= 12 &&
    data[0] === 0x52 &&
    data[1] === 0x49 &&
    data[2] === 0x46 &&
    data[3] === 0x46 &&
    data[8] === 0x57 &&
    data[9] === 0x45 &&
    data[10] === 0x42 &&
    data[11] === 0x50
  ) {
    return 'image/webp';
  }
  if (
    data.length >= 8 &&
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a
  ) {
    return 'image/png';
  }
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    data.length >= 6 &&
    data[0] === 0x47 &&
    data[1] === 0x49 &&
    data[2] === 0x46 &&
    data[3] === 0x38 &&
    (data[4] === 0x37 || data[4] === 0x39) &&
    data[5] === 0x61
  ) {
    return 'image/gif';
  }
  return null;
}

function extractImagePayload(data: Uint8Array): ImagePayload | null {
  const mime = detectImageMime(data);
  if (!mime) return null;

  if (mime === 'image/webp' && data.length >= 8) {
    const riffSize = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24);
    const total = Math.min(data.length, (riffSize >>> 0) + 8);
    return { mime, bytes: data.slice(0, total) };
  }

  return { mime, bytes: trimTrailingZeros(data) };
}

function setDecodedOutput(recovered: Uint8Array) {
  clearOutput();
  const image = extractImagePayload(recovered);
  if (image) {
    outputImageMime.value = image.mime;
    const blobBytes = new Uint8Array(image.bytes.length);
    blobBytes.set(image.bytes);
    outputImageUrl.value = URL.createObjectURL(
      new Blob([blobBytes.buffer], { type: image.mime })
    );
    return;
  }
  outputText.value = new TextDecoder().decode(trimTrailingZeros(recovered));
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
  safeDisconnect(rxInputGain, decoderNode);
  safeDisconnect(rxInputGain);
  safeDisconnect(decoderNode, decoderStreamSink);
  safeDisconnect(decoderNode);

  micSource = null;
  rxInputGain = null;
  decoderNode = null;
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
    makeOnProgressCallback()
  );
}

async function toggleMic() {
  await runExclusive(async () => {
    if (!runtime.coreReady.value) return;
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
        <h2>Receiver</h2>
        <p class="panel-sub">Adaptive K decode + progress tracing</p>
        <div class="status-chip" :class="receiverStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
          {{ receiverStatus }}
        </div>
      </div>
      <div class="button-row compact">
        <button @click="toggleMic" :class="{ 'btn-active': isMicActive }" class="btn" :disabled="!runtime.coreReady.value">
          {{ isMicActive ? 'Disable Mic' : 'Enable Mic' }}
        </button>
        <button @click="reset" class="btn" :disabled="!runtime.coreReady.value">Clear</button>
      </div>
    </div>

    <div class="path-banner">
      <span class="path-label">Input Path</span>
      <code v-if="!isMicActive">[demoAirGapNode] -digital- [Receiver]</code>
      <code v-else>[Mic] -acoustic- [Receiver]</code>
    </div>

    <div class="display">
      <p class="display-title">Decoded Result</p>
      <pre v-if="outputText">{{ outputText }}</pre>
      <div v-else-if="outputImageUrl" class="image-result">
        <img :src="outputImageUrl" :alt="`decoded image (${outputImageMime || 'unknown'})`" />
        <p class="image-meta">{{ outputImageMime }}</p>
      </div>
      <p v-else class="placeholder">Waiting for synchronization...</p>
    </div>

    <div class="progress-block">
      <div class="progress-head">
        <span>Rank {{ rankPackets }} / {{ totalNeededPackets || '?' }}</span>
        <span>{{ (progressPercent * 100).toFixed(1) }}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" :style="{ width: `${progressPercent * 100}%` }" />
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric"><span>Accepted</span><strong>{{ receivedPackets }}</strong></div>
      <div class="metric"><span>Stall</span><strong>{{ stalledPackets }}</strong></div>
      <div class="metric"><span>Dep</span><strong>{{ dependentPackets }}</strong></div>
      <div class="metric"><span>Dup</span><strong>{{ duplicatePackets }}</strong></div>
      <div class="metric"><span>CRC</span><strong>{{ crcErrorPackets }}</strong></div>
      <div class="metric"><span>Parse</span><strong>{{ parseErrorPackets }}</strong></div>
      <div class="metric"><span>InvNbr</span><strong>{{ invalidNeighborPackets }}</strong></div>
      <div class="metric"><span>Last Seq</span><strong>{{ lastPacketSeq }}</strong></div>
      <div class="metric"><span>Last RankUp</span><strong>{{ lastRankUpSeq }}</strong></div>
    </div>

    <div class="proc-stats">
      <p class="proc-title">DecoderProcessor Timing</p>
      <div class="proc-grid">
        <div><span>avg</span><strong>{{ decoderProcAvgMs.toFixed(3) }} ms</strong></div>
        <div><span>max</span><strong>{{ decoderProcMaxMs.toFixed(3) }} ms</strong></div>
        <div><span>last</span><strong>{{ decoderProcLastMs.toFixed(3) }} ms</strong></div>
        <div><span>budget</span><strong>{{ decoderProcBlockMs.toFixed(3) }} ms</strong></div>
        <div><span>overrun</span><strong>{{ decoderProcOverruns }}</strong></div>
        <div><span>input RMS</span><strong>{{ decoderProcInputRms.toFixed(5) }}</strong></div>
        <div><span>blocks</span><strong>{{ decoderProcBlocks }}</strong></div>
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
