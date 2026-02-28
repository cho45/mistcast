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

  decoderNode = new AudioWorkletNode(audioContext, 'decoder-processor');
  await backend.setAudioInPort(Comlink.transfer(decoderNode.port, [decoderNode.port]));

  encoderNode.connect(audioContext.destination);
  encoderNode.connect(decoderNode);
  
  status.value = "Ready";
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

  // デコーダはパケット内のk情報から自動追従する。
  await backend.startDecoder(
    audioContext.sampleRate, 
    Comlink.proxy((recovered: Uint8Array) => {
        setDecodedOutput(recovered);
        status.value = "Decoded!";
    }),
    Comlink.proxy((p: any) => {
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
    })
  );

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
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } 
      });
      micSource = audioContext.createMediaStreamSource(stream);
      encoderNode.disconnect(decoderNode);
      micSource.connect(decoderNode);
      isMicActive.value = true;
      status.value = "Mic Active";
    } catch (e) {
      console.error(e);
      status.value = "Mic Error";
    }
  } else {
    micSource?.disconnect();
    micSource = null;
    encoderNode.connect(decoderNode);
    isMicActive.value = false;
    status.value = "Internal Loopback";
  }
}

async function reset() {
    await backend?.resetDecoder();
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
    status.value = "Ready";
}

onBeforeUnmount(() => {
  clearOutput();
});
</script>

<template>
  <div class="container">
    <h1>Mistcast Demo v2.3</h1>
    
    <div class="card">
      <div class="status-bar" :class="status.toLowerCase().replace(' ', '-')">
        Status: {{ status }}
      </div>
      
      <button v-if="!backend" @click="init" class="btn btn-large">Initialize Audio System</button>
      
      <div v-if="backend" class="controls">
        <div class="send-section">
          <h3>Sender (Fountain Stream)</h3>
          <textarea v-model="inputText" rows="3" placeholder="Enter text to broadcast..."></textarea>
          <div class="btn-group">
            <button @click="startSendingText" class="btn btn-primary" :disabled="status === 'Transmitting...'">Send Text</button>
            <button @click="startSendingSampleImage" class="btn" :disabled="status === 'Transmitting...'">Send Sample Image</button>
            <button @click="stopSending" class="btn">Stop</button>
          </div>
        </div>

        <div class="recv-section">
          <h3>Receiver (Adaptive K)</h3>
          <div class="btn-group">
            <button @click="toggleMic" :class="{ 'btn-active': isMicActive }" class="btn">
              {{ isMicActive ? 'Disable Mic' : 'Enable Microphone' }}
            </button>
            <button @click="reset" class="btn">Clear</button>
          </div>
          
          <div class="progress-container" v-if="totalNeededPackets > 0">
            <div class="progress-text">
              Rank: {{ rankPackets }} / {{ totalNeededPackets }} ({{ (progressPercent * 100).toFixed(1) }}%)
            </div>
            <div class="progress-text detail">
              Accepted: {{ receivedPackets }}
              | Stall: {{ stalledPackets }}
              | Dep: {{ dependentPackets }}
              | Dup: {{ duplicatePackets }}
              | CRC: {{ crcErrorPackets }}
              | Parse: {{ parseErrorPackets }}
              | InvNbr: {{ invalidNeighborPackets }}
              | Last Seq: {{ lastPacketSeq }}
              | Last Rank-Up Seq: {{ lastRankUpSeq }}
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" :style="{ width: (progressPercent * 100) + '%' }"></div>
            </div>
            <div class="rx-log" v-if="rxLogs.length > 0">
              <pre>{{ rxLogs.join('\n') }}</pre>
            </div>
          </div>

          <div class="display">
            <p><strong>Decoded Result:</strong></p>
            <pre v-if="outputText">{{ outputText }}</pre>
            <div v-else-if="outputImageUrl" class="image-result">
              <img :src="outputImageUrl" :alt="`decoded image (${outputImageMime || 'unknown'})`" />
              <p class="image-meta">{{ outputImageMime }}</p>
            </div>
            <p v-else class="placeholder">Waiting for synchronization...</p>
          </div>
        </div>
      </div>
    </div>

    <div class="info">
      <h4>Connection Path:</h4>
      <p v-if="!isMicActive"><code>[Encoder] --(Digital)--> [Decoder]</code> (Internal Loopback)</p>
      <p v-else><code>[Mic Input] --(Analog)--> [Decoder]</code> (Acoustic Air-gap)</p>
      <p>Speaker Output: <code>[Encoder] --(Sound)--> [Destination]</code></p>
    </div>
  </div>
</template>

<style>
.container { max-width: 900px; margin: 0 auto; padding: 2rem; font-family: sans-serif; color: #333; }
.card { border: 1px solid #ddd; padding: 2rem; border-radius: 12px; background: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.status-bar { padding: 0.5rem 1rem; border-radius: 4px; margin-bottom: 1.5rem; font-weight: bold; background: #eee; text-align: center; }
.status-bar.transmitting { background: #fff3cd; color: #856404; }
.status-bar.decoded { background: #d4edda; color: #155724; }
.status-bar.ready { background: #d1ecf1; color: #0c5460; }
.controls { display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; }
textarea { width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; margin-bottom: 1rem; }
.btn-group { display: flex; gap: 0.5rem; }
.progress-container { margin-top: 1.5rem; }
.progress-text { font-size: 0.85rem; margin-bottom: 0.4rem; color: #666; }
.progress-text.detail { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #555; }
.progress-bar-bg { width: 100%; height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }
.progress-bar-fill { height: 100%; background: #28a745; transition: width 0.2s ease-out; }
.rx-log { margin-top: 0.6rem; max-height: 220px; overflow: auto; border: 1px solid #e0e0e0; border-radius: 4px; background: #fafafa; }
.rx-log pre { margin: 0; padding: 0.6rem; font-size: 0.78rem; line-height: 1.35; color: #444; }
.display { margin-top: 1rem; background: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px; min-height: 120px; }
.placeholder { color: #999; font-style: italic; }
pre { white-space: pre-wrap; font-size: 1.1rem; color: #007bff; margin: 0; }
.image-result { display: flex; flex-direction: column; gap: 0.5rem; }
.image-result img { max-width: 100%; max-height: 240px; object-fit: contain; border: 1px solid #ddd; border-radius: 4px; background: #fff; }
.image-meta { margin: 0; color: #666; font-size: 0.85rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
button { padding: 0.6rem 1.2rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-weight: 600; }
.btn-primary { background: #007bff; color: #fff; }
.btn-active { background: #dc3545; color: #fff; }
.btn-large { width: 100%; font-size: 1.2rem; padding: 1rem; }
.info { margin-top: 2rem; font-size: 0.9rem; color: #666; background: #f1f1f1; padding: 1rem; border-radius: 8px; }
code { background: #e0e0e0; padding: 0.2rem 0.4rem; border-radius: 3px; }
</style>
