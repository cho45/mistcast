<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import sampleFileUrl from '../assets/sample-files/test.png';
import { useDemoRuntime } from '../demo-runtime';

const runtime = useDemoRuntime();

const inputText = ref('Hello Acoustic World!');
const fftCanvas = ref<HTMLCanvasElement | null>(null);
const isTransmitting = ref(false);
const senderStatus = ref('Idle');

let senderWorker: Worker | null = null;
let senderBackend: Comlink.Remote<MistcastBackend> | null = null;
let encoderNode: AudioWorkletNode | null = null;
let analyserNode: AnalyserNode | null = null;
let fftData: Float32Array<ArrayBuffer> | null = null;
let fftRafId: number | null = null;

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

function stopSenderSpectrum() {
  if (fftRafId !== null) {
    window.cancelAnimationFrame(fftRafId);
    fftRafId = null;
  }
}

function drawSenderSpectrumFrame(audioContext: AudioContext) {
  if (!analyserNode || !fftData || !fftCanvas.value) {
    fftRafId = window.requestAnimationFrame(() => drawSenderSpectrumFrame(audioContext));
    return;
  }

  const canvas = fftCanvas.value;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    fftRafId = window.requestAnimationFrame(() => drawSenderSpectrumFrame(audioContext));
    return;
  }

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(10, canvas.clientWidth || 640);
  const cssH = Math.max(10, canvas.clientHeight || 180);
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  analyserNode.getFloatFrequencyData(fftData);
  const nyquist = audioContext.sampleRate / 2;
  const fMax = Math.max(1, Math.min(20000, nyquist));
  const minDb = analyserNode.minDecibels;
  const maxDb = analyserNode.maxDecibels;

  const xFromFreq = (f: number) => (Math.max(0, Math.min(f, fMax)) / fMax) * cssW;
  const yFromDb = (db: number) => {
    const n = (db - minDb) / (maxDb - minDb);
    return cssH - Math.min(1, Math.max(0, n)) * cssH;
  };

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = '#f9fcff';
  ctx.fillRect(0, 0, cssW, cssH);

  ctx.strokeStyle = '#d8e3ef';
  ctx.lineWidth = 1;
  const tickStep = fMax <= 12000 ? 1000 : 2000;
  const freqTicks: number[] = [];
  for (let f = 0; f <= fMax; f += tickStep) freqTicks.push(f);
  for (const f of freqTicks) {
    const x = xFromFreq(f);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
  }

  const dbTicks = [-90, -75, -60, -45, -30, -15, 0];
  for (const db of dbTicks) {
    const y = yFromDb(db);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cssW, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#0f6bd7';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const bins = fftData.length;
  for (let x = 0; x < cssW; x++) {
    const t = x / Math.max(1, cssW - 1);
    const f = t * fMax;
    const bin = Math.min(bins - 1, Math.max(0, Math.round((f / nyquist) * (bins - 1))));
    const y = yFromDb(fftData[bin]);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#5a6470';
  ctx.font = '11px IBM Plex Mono, Menlo, monospace';
  for (const f of freqTicks) {
    const x = xFromFreq(f);
    const label = f >= 1000 ? `${Math.round(f / 1000)}k` : `${f}`;
    ctx.fillText(label, Math.min(cssW - 22, Math.max(0, x + 2)), cssH - 4);
  }

  fftRafId = window.requestAnimationFrame(() => drawSenderSpectrumFrame(audioContext));
}

function startSenderSpectrum(audioContext: AudioContext) {
  if (fftRafId !== null) return;
  fftRafId = window.requestAnimationFrame(() => drawSenderSpectrumFrame(audioContext));
}

async function loadSampleFile(): Promise<Uint8Array> {
  const res = await fetch(sampleFileUrl);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

async function teardownSenderGraph() {
  stopSenderSpectrum();

  if (senderBackend) {
    try {
      await senderBackend.stopEncoder();
    } catch {
      // no-op
    }
  }

  if (senderWorker) {
    senderWorker.terminate();
    senderWorker = null;
  }
  senderBackend = null;

  safeDisconnect(encoderNode, analyserNode);
  safeDisconnect(encoderNode);
  safeDisconnect(analyserNode);

  encoderNode = null;
  analyserNode = null;
  fftData = null;
}

async function rebuildSenderGraph() {
  const { audioContext, demoAirGapNode } = await runtime.ensureAudioCore();
  await teardownSenderGraph();

  senderWorker = new MistcastWorker();
  senderBackend = Comlink.wrap<MistcastBackend>(senderWorker);
  await senderBackend.init();

  encoderNode = new AudioWorkletNode(audioContext, 'encoder-processor');
  await senderBackend.setAudioOutPort(Comlink.transfer(encoderNode.port, [encoderNode.port]));

  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 4096;
  analyserNode.smoothingTimeConstant = 0.6;
  analyserNode.minDecibels = -100;
  analyserNode.maxDecibels = -20;
  fftData = new Float32Array(
    new ArrayBuffer(analyserNode.frequencyBinCount * Float32Array.BYTES_PER_ELEMENT)
  );

  encoderNode.connect(audioContext.destination);
  encoderNode.connect(analyserNode);
  encoderNode.connect(demoAirGapNode);
  startSenderSpectrum(audioContext);
}

async function startSendingData(data: Uint8Array) {
  await runExclusive(async () => {
    if (!runtime.coreReady.value) return;
    const { audioContext } = await runtime.ensureAudioCore();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    senderStatus.value = 'Preparing...';
    await rebuildSenderGraph();
    if (!senderBackend) return;

    senderStatus.value = 'Transmitting...';
    isTransmitting.value = true;
    await senderBackend.startEncoder(data, audioContext.sampleRate);
  });
}

async function startSendingText() {
  const data = new TextEncoder().encode(inputText.value);
  await startSendingData(data);
}

async function startSendingSampleImage() {
  const data = await loadSampleFile();
  await startSendingData(data);
}

async function stopSending() {
  await runExclusive(async () => {
    await senderBackend?.stopEncoder();
    isTransmitting.value = false;
    senderStatus.value = runtime.coreReady.value ? 'Ready' : 'Idle';
  });
}

watch(
  () => runtime.coreReady.value,
  async (ready) => {
    if (!ready) return;
    await runExclusive(async () => {
      await rebuildSenderGraph();
      senderStatus.value = 'Ready';
    });
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  void teardownSenderGraph();
});
</script>

<template>
  <section class="panel sender-panel">
    <h2>Sender</h2>
    <p class="panel-sub">Text / Image を音響フレームへ変調して送信</p>
    <div class="status-chip" :class="senderStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
      {{ senderStatus }}
    </div>
    <textarea v-model="inputText" rows="4" placeholder="Enter text to broadcast..." />
    <div class="button-row">
      <button @click="startSendingText" class="btn btn-primary" :disabled="!runtime.coreReady.value || isTransmitting">Send Text</button>
      <button @click="startSendingSampleImage" class="btn" :disabled="!runtime.coreReady.value || isTransmitting">Send Sample Image</button>
      <button @click="stopSending" class="btn btn-danger" :disabled="!runtime.coreReady.value">Stop</button>
    </div>
    <div class="spectrum-panel">
      <p class="spectrum-title">Sender FFT (Linear Frequency Axis)</p>
      <canvas ref="fftCanvas" class="spectrum-canvas"></canvas>
    </div>
  </section>
</template>
