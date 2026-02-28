<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from './worker';
import sampleWebpUrl from './assets/sample-files/webp.webp';

// Vite's worker loading
import MistcastWorker from './worker?worker';
// AudioWorklet module URL
import processorsUrl from './audio-processors?url';

const inputText = ref("Hello Acoustic World!");
const outputText = ref("");
const outputImageUrl = ref("");
const outputImageMime = ref("");
const status = ref("Idle");
const isMicActive = ref(false);

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
const rxLogs = ref<string[]>([]);
const rxTick = ref(0);

let backend: Comlink.Remote<MistcastBackend> | null = null;
let audioContext: AudioContext | null = null;
let encoderNode: AudioWorkletNode | null = null;
let decoderNode: AudioWorkletNode | null = null;
let micSource: MediaStreamAudioSourceNode | null = null;
let micStream: MediaStream | null = null;

type ImagePayload = {
  mime: string;
  bytes: Uint8Array;
};

function clearOutput() {
  outputText.value = "";
  outputImageMime.value = "";
  if (outputImageUrl.value) {
    URL.revokeObjectURL(outputImageUrl.value);
    outputImageUrl.value = "";
  }
}

function trimTrailingZeros(data: Uint8Array): Uint8Array {
  let last = data.length;
  while (last > 0 && data[last - 1] === 0) last--;
  return data.slice(0, last);
}

function detectImageMime(data: Uint8Array): string | null {
  if (data.length >= 12 &&
      data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46 &&
      data[8] === 0x57 && data[9] === 0x45 && data[10] === 0x42 && data[11] === 0x50) {
    return "image/webp";
  }
  if (data.length >= 8 &&
      data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47 &&
      data[4] === 0x0d && data[5] === 0x0a && data[6] === 0x1a && data[7] === 0x0a) {
    return "image/png";
  }
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return "image/jpeg";
  }
  if (data.length >= 6 &&
      data[0] === 0x47 && data[1] === 0x49 && data[2] === 0x46 &&
      data[3] === 0x38 && (data[4] === 0x37 || data[4] === 0x39) && data[5] === 0x61) {
    return "image/gif";
  }
  return null;
}

function extractImagePayload(data: Uint8Array): ImagePayload | null {
  const mime = detectImageMime(data);
  if (!mime) return null;

  if (mime === "image/webp" && data.length >= 8) {
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
    outputImageUrl.value = URL.createObjectURL(new Blob([blobBytes.buffer], { type: image.mime }));
    return;
  }
  outputText.value = new TextDecoder().decode(trimTrailingZeros(recovered));
}

async function loadSampleWebp(): Promise<Uint8Array> {
  const res = await fetch(sampleWebpUrl);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

function pushRxLog(line: string) {
  rxLogs.value.push(line);
  if (rxLogs.value.length > 120) {
    rxLogs.value.splice(0, rxLogs.value.length - 120);
  }
}

function makeOnPacketCallback() {
  return Comlink.proxy((recovered: Uint8Array) => {
    setDecodedOutput(recovered);
    status.value = "Decoded!";
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
    rxTick.value += 1;
    const changed = prevReceived !== receivedPackets.value || prevRank !== rankPackets.value || p.complete;
    if (changed) {
      pushRxLog(
        `#${rxTick.value} recv=${receivedPackets.value} rank=${rankPackets.value}/${totalNeededPackets.value} stall=${stalledPackets.value} dup=${duplicatePackets.value} crc=${crcErrorPackets.value} parse=${parseErrorPackets.value} invN=${invalidNeighborPackets.value} prog=${(progressPercent.value * 100).toFixed(1)}% lastSeq=${lastPacketSeq.value} lastRankUp=${lastRankUpSeq.value}${p.complete ? " COMPLETE" : ""}`
      );
    }
  });
}

async function startDecoderStandby() {
  if (!backend || !audioContext) return;
  await backend.startDecoder(
    audioContext.sampleRate,
    makeOnPacketCallback(),
    makeOnProgressCallback()
  );
}

async function init() {
  if (backend) return;
  status.value = "Initializing...";

  const worker = new MistcastWorker();
  backend = Comlink.wrap<MistcastBackend>(worker);
  await backend.init();

  audioContext = new AudioContext({ sampleRate: 48000 });
  await audioContext.audioWorklet.addModule(processorsUrl);

  encoderNode = new AudioWorkletNode(audioContext, 'encoder-processor');
  await backend.setAudioOutPort(Comlink.transfer(encoderNode.port, [encoderNode.port]));

  decoderNode = new AudioWorkletNode(audioContext, 'decoder-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1,
    channelCountMode: "explicit",
    channelInterpretation: "discrete",
  });
  await backend.setAudioInPort(Comlink.transfer(decoderNode.port, [decoderNode.port]));

  encoderNode.connect(audioContext.destination);
  encoderNode.connect(decoderNode);

  await startDecoderStandby();
  
  status.value = "Ready (Rx standby)";
}

async function startSendingData(data: Uint8Array) {
  if (!backend || !audioContext) return;
  if (audioContext.state === 'suspended') await audioContext.resume();

  status.value = "Preparing...";
  clearOutput();
  receivedPackets.value = 0;
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
  rxLogs.value = [];
  rxTick.value = 0;

  // デコーダは常時待機させる（受信専用機でも動作させるため）。
  await startDecoderStandby();

  status.value = "Transmitting...";
  await backend.startEncoder(data, audioContext.sampleRate);
}

async function startSendingText() {
  const data = new TextEncoder().encode(inputText.value);
  await startSendingData(data);
}

async function startSendingSampleImage() {
  const data = await loadSampleWebp();
  await startSendingData(data);
}

async function stopSending() {
    await backend?.stopEncoder();
    status.value = "Stopped";
}

async function toggleMic() {
  if (!audioContext || !decoderNode || !encoderNode) return;
  if (audioContext.state === 'suspended') await audioContext.resume();

  if (!isMicActive.value) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        } 
      });
      micStream = stream;
      micSource = audioContext.createMediaStreamSource(stream);
      encoderNode.disconnect(decoderNode);
      micSource.connect(decoderNode);
      isMicActive.value = true;
      status.value = "Mic Active (Rx)";
    } catch (e) {
      console.error(e);
      status.value = "Mic Error";
    }
  } else {
    micSource?.disconnect();
    micSource = null;
    micStream?.getTracks().forEach((t) => t.stop());
    micStream = null;
    encoderNode.connect(decoderNode);
    isMicActive.value = false;
    status.value = "Internal Loopback";
  }
}

async function reset() {
    await backend?.resetDecoder();
    await startDecoderStandby();
    clearOutput();
    receivedPackets.value = 0;
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
    totalNeededPackets.value = 0;
    rxLogs.value = [];
    rxTick.value = 0;
    status.value = "Ready (Rx standby)";
}

onBeforeUnmount(() => {
  micSource?.disconnect();
  micStream?.getTracks().forEach((t) => t.stop());
  clearOutput();
});
</script>

<template>
  <div class="app-shell">
    <header class="hero">
      <h1>Mistcast Demo v2.3</h1>
      <p>Acoustic DSSS + RLNC playground</p>
      <div class="status-chip" :class="status.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
        {{ status }}
      </div>
    </header>

    <main class="content">
      <section v-if="!backend" class="panel init-panel">
        <p>まず Audio System を初期化して、送受信ノードを作成します。</p>
        <button @click="init" class="btn btn-primary btn-large">Initialize Audio System</button>
      </section>

      <template v-else>
        <div class="split-panels">
          <section class="panel sender-panel">
            <h2>Sender</h2>
            <p class="panel-sub">Text / Image を音響フレームへ変調して送信</p>
            <textarea v-model="inputText" rows="4" placeholder="Enter text to broadcast..." />
            <div class="button-row">
              <button @click="startSendingText" class="btn btn-primary" :disabled="status === 'Transmitting...'">Send Text</button>
              <button @click="startSendingSampleImage" class="btn" :disabled="status === 'Transmitting...'">Send Sample Image</button>
              <button @click="stopSending" class="btn btn-danger">Stop</button>
            </div>
          </section>

          <section class="panel receiver-panel">
            <div class="receiver-header">
              <div>
                <h2>Receiver</h2>
                <p class="panel-sub">Adaptive K decode + progress tracing</p>
              </div>
              <div class="button-row compact">
                <button @click="toggleMic" :class="{ 'btn-active': isMicActive }" class="btn">
                  {{ isMicActive ? 'Disable Mic' : 'Enable Mic' }}
                </button>
                <button @click="reset" class="btn">Clear</button>
              </div>
            </div>

            <div class="path-banner">
              <span class="path-label">Input Path</span>
              <code v-if="!isMicActive">[Encoder] -digital- [Decoder]</code>
              <code v-else>[Mic] -acoustic- [Decoder]</code>
            </div>

            <div class="progress-block">
              <div class="progress-head">
                <span>Rank {{ rankPackets }} / {{ totalNeededPackets || "?" }}</span>
                <span>{{ (progressPercent * 100).toFixed(1) }}%</span>
              </div>
              <div class="progress-bar-bg">
                <div class="progress-bar-fill" :style="{ width: (progressPercent * 100) + '%' }" />
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

            <div class="rx-log" v-if="rxLogs.length > 0">
              <pre>{{ rxLogs.join('\n') }}</pre>
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
          </section>
        </div>
      </template>
    </main>

    <footer class="footnote">
      <div><strong>Speaker Out:</strong> <code>[Encoder] -sound- [Destination]</code></div>
    </footer>
  </div>
</template>

<style>
* {
  box-sizing: border-box;
}

:root {
  --bg-a: #f7f5ee;
  --bg-b: #e3ebf3;
  --panel: #ffffff;
  --ink: #1b2229;
  --muted: #5a6470;
  --line: #d8dee8;
  --primary: #0f6bd7;
  --primary-strong: #004fb3;
  --danger: #b94731;
  --good: #1a8f55;
  --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: "Avenir Next", "Segoe UI", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
}

.app-shell {
  min-height: 100vh;
  margin: 0 auto;
  padding: 1rem;
  color: var(--ink);
  overflow-x: hidden;
  background:
    radial-gradient(120% 120% at 0% 0%, #fff6da 0%, transparent 45%),
    radial-gradient(130% 120% at 100% 0%, #d9efff 0%, transparent 45%),
    linear-gradient(165deg, var(--bg-a), var(--bg-b));
  font-family: var(--sans);
}

.hero {
  max-width: 1100px;
  margin: 0 auto 1rem;
  padding: 0.25rem 0.1rem;
}

.hero h1 {
  margin: 0;
  font-size: clamp(1.35rem, 2.3vw, 2rem);
  letter-spacing: 0.01em;
}

.hero p {
  margin: 0.25rem 0 0.8rem;
  color: var(--muted);
  font-size: 0.92rem;
}

.status-chip {
  display: inline-block;
  padding: 0.4rem 0.7rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #fff;
  font-weight: 700;
  font-size: 0.82rem;
}

.status-chip.transmitting {
  color: #8e5a00;
  background: #fff4d8;
  border-color: #f0d299;
}

.status-chip.decoded {
  color: #0d6b3d;
  background: #def8ea;
  border-color: #9cd6b8;
}

.status-chip.ready-rx-standby,
.status-chip.ready {
  color: #0a557f;
  background: #e4f2fb;
  border-color: #a9cfe6;
}

.content {
  max-width: 1100px;
  margin: 0 auto;
}

.split-panels {
  display: grid;
  gap: 1rem;
  grid-template-columns: 1fr;
  min-width: 0;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(16, 24, 40, 0.07);
  min-width: 0;
}

.panel h2 {
  margin: 0;
  font-size: 1.06rem;
}

.panel-sub {
  margin: 0.2rem 0 0.9rem;
  color: var(--muted);
  font-size: 0.84rem;
}

.init-panel {
  text-align: center;
}

textarea {
  width: 100%;
  min-height: 110px;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.7rem;
  font-size: 0.95rem;
  font-family: var(--sans);
  background: #fbfdff;
  color: var(--ink);
}

.button-row {
  margin-top: 0.75rem;
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.5rem;
}

.button-row.compact {
  margin-top: 0;
}

.btn {
  border: 1px solid #b8c4d1;
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  background: #fff;
  color: var(--ink);
  font-weight: 700;
  cursor: pointer;
  font-size: 0.9rem;
}

.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.btn-primary {
  color: #fff;
  border-color: var(--primary-strong);
  background: linear-gradient(180deg, var(--primary), var(--primary-strong));
}

.btn-danger {
  border-color: #d8b2a9;
  color: #7e2d1f;
  background: #fdebe7;
}

.btn-active {
  color: #fff;
  border-color: #8f2514;
  background: #be341e;
}

.btn-large {
  width: 100%;
  padding: 0.9rem 1rem;
  font-size: 1rem;
}

.receiver-header {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.path-banner {
  margin-top: 0.3rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0.55rem 0.7rem;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: #f7fafc;
  min-width: 0;
}

.path-label {
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 700;
}

code {
  font-family: var(--mono);
  background: #eef3f8;
  color: #24455e;
  border: 1px solid #d6e1ea;
  border-radius: 8px;
  padding: 0.2rem 0.35rem;
  display: inline-block;
  max-width: 100%;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.progress-block {
  margin-top: 0.9rem;
}

.progress-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.84rem;
  color: var(--muted);
  margin-bottom: 0.35rem;
  min-width: 0;
}

.progress-head span {
  min-width: 0;
  overflow-wrap: anywhere;
}

.progress-bar-bg {
  width: 100%;
  height: 10px;
  background: #e8edf3;
  border-radius: 999px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #2ba463, var(--good));
  transition: width 0.2s ease-out;
}

.metric-grid {
  margin-top: 0.9rem;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.5rem;
}

.metric {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.45rem 0.55rem;
  background: #fcfdff;
  min-height: 54px;
}

.metric span {
  display: block;
  color: var(--muted);
  font-size: 0.72rem;
  margin-bottom: 0.15rem;
}

.metric strong {
  font-family: var(--mono);
  font-size: 0.95rem;
}

.rx-log {
  margin-top: 0.85rem;
  max-height: 220px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfcfe;
}

.rx-log pre {
  margin: 0;
  padding: 0.6rem;
  font-size: 0.73rem;
  line-height: 1.4;
  font-family: var(--mono);
  color: #2f475a;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.display {
  margin-top: 0.95rem;
  background: #f6f9fc;
  border: 1px solid var(--line);
  padding: 0.75rem;
  border-radius: 10px;
  min-height: 140px;
  min-width: 0;
}

.display-title {
  margin: 0 0 0.5rem;
  color: #314453;
  font-size: 0.88rem;
  font-weight: 700;
}

.display pre {
  margin: 0;
  white-space: pre-wrap;
  font-size: 0.96rem;
  color: #0c63bd;
}

.placeholder {
  margin: 0;
  color: #738090;
  font-style: italic;
  font-size: 0.9rem;
}

.image-result {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.image-result img {
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  border: 1px solid #cfdae6;
  border-radius: 8px;
  background: #fff;
}

.image-meta {
  margin: 0;
  color: var(--muted);
  font-size: 0.8rem;
  font-family: var(--mono);
}

.footnote {
  max-width: 1100px;
  margin: 1rem auto 0;
  color: var(--muted);
  font-size: 0.84rem;
  padding: 0.35rem 0.1rem 0.9rem;
}

@media (min-width: 780px) {
  .app-shell {
    padding: 1.6rem;
  }

  .split-panels {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }

  .receiver-header {
    flex-direction: row;
    justify-content: space-between;
    align-items: flex-start;
  }

  .button-row {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .button-row.compact {
    grid-template-columns: repeat(2, minmax(0, max-content));
  }

  .metric-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}
</style>
