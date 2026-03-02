<script setup lang="ts">
import { onBeforeUnmount, ref, useAttrs, watch } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import sampleFileUrl from '../assets/sample-files/test.png';
import { useDemoRuntime } from '../demo-runtime';

// Disable automatic attribute inheritance since we have multiple root nodes
defineOptions({
  inheritAttrs: false,
});

const runtime = useDemoRuntime();
const attrs = useAttrs();

const MAX_FILE_SIZE = 255 * 16; // 4080 bytes
const TOAST_DURATION_MS = 5000;

type ToastType = 'error' | 'warning' | 'success';

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

const inputText = ref('Hello Acoustic World!');
const fftCanvas = ref<HTMLCanvasElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const isTransmitting = ref(false);
const senderStatus = ref('Idle');
const isDragging = ref(false);
const toasts = ref<Toast[]>([]);
let toastIdCounter = 0;

let senderWorker: Worker | null = null;
let senderBackend: Comlink.Remote<MistcastBackend> | null = null;
let encoderNode: AudioWorkletNode | null = null;
let analyserNode: AnalyserNode | null = null;
let fftData: Float32Array<ArrayBuffer> | null = null;
let fftRafId: number | null = null;

let opQueue: Promise<void> = Promise.resolve();

function validateFileSize(size: number): boolean {
  return size <= MAX_FILE_SIZE;
}

function showToast(message: string, type: ToastType = 'error') {
  const id = toastIdCounter++;
  toasts.value.push({ id, message, type });
  setTimeout(() => {
    toasts.value = toasts.value.filter(t => t.id !== id);
  }, TOAST_DURATION_MS);
}

function triggerFileSelect() {
  fileInput.value?.click();
}

async function startSendingFile(file: File) {
  const buf = await file.arrayBuffer();
  const data = new Uint8Array(buf);
  await startSendingData(data);
}

async function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (file) {
    if (!validateFileSize(file.size)) {
      showToast(`ファイルサイズが大きすぎます（最大 ${MAX_FILE_SIZE} バイト）`);
      target.value = '';
      return;
    }
    await startSendingFile(file);
  }
  // input要素の値をリセットして同じファイルを再度選択可能にする
  target.value = '';
}

function handleDragEnter(event: DragEvent) {
  event.preventDefault();
  // 子要素からの dragenter を無視
  if (event.target !== event.currentTarget) return;
  isDragging.value = true;
}

function handleDragLeave(event: DragEvent) {
  event.preventDefault();
  // 子要素に入っただけなら無視（本当に要素から出たかチェック）
  const target = event.currentTarget as HTMLElement;
  if (!target) return;
  const relatedTarget = event.relatedTarget as HTMLElement | null;
  // relatedTarget が null（ウィンドウ外に出た）または、現在の要素の子孫でない場合のみ true
  if (relatedTarget && target.contains(relatedTarget)) return;
  isDragging.value = false;
}

function handleDragOver(event: DragEvent) {
  event.preventDefault();
}

async function handleDrop(event: DragEvent) {
  event.preventDefault();
  isDragging.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    if (!validateFileSize(file.size)) {
      showToast(`ファイルサイズが大きすぎます（最大 ${MAX_FILE_SIZE} バイト）`);
      return;
    }
    await startSendingFile(file);
  }
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
  if (!validateFileSize(data.length)) {
    showToast(`テキストサイズが大きすぎます（最大 ${MAX_FILE_SIZE} バイト）`);
    return;
  }
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

defineExpose({
  toasts,
});
</script>

<template>
  <section
    class="panel sender-panel"
    :class="{ 'is-dragging': isDragging }"
    v-bind="attrs"
    @dragenter="handleDragEnter"
    @dragleave="handleDragLeave"
    @dragover="handleDragOver"
    @drop="handleDrop"
  >
    <h2>Sender</h2>
    <p class="panel-sub">Text / Image を音響フレームへ変調して送信</p>
    <div class="status-chip" :class="senderStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
      {{ senderStatus }}
    </div>
    <textarea v-model="inputText" rows="4" placeholder="Enter text to broadcast..." />
    <div class="button-row">
      <template v-if="!isTransmitting">
        <button @click="startSendingText" class="btn btn-primary" :disabled="!runtime.coreReady.value">Send</button>
        <button @click="startSendingSampleImage" class="btn" :disabled="!runtime.coreReady.value">Send Sample Image</button>
        <button @click="triggerFileSelect" class="btn" :disabled="!runtime.coreReady.value">Send File (max 4KB)</button>
        <input type="file" ref="fileInput" style="display: none" @change="handleFileSelect" />
      </template>
      <button v-else @click="stopSending" class="btn btn-danger" :disabled="!runtime.coreReady.value">Stop</button>
    </div>
    <div class="spectrum-panel">
      <p class="spectrum-title">Sender FFT (Linear Frequency Axis)</p>
      <canvas ref="fftCanvas" class="spectrum-canvas"></canvas>
    </div>
  </section>

  <Teleport to="body">
    <div class="toast-container">
      <TransitionGroup name="toast">
        <div v-for="toast in toasts" :key="toast.id" class="toast" :class="`toast-${toast.type}`">
          {{ toast.message }}
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 9999;
  pointer-events: none;
}

.toast {
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  pointer-events: auto;
  max-width: 350px;
  word-wrap: break-word;
}

.toast-error {
  background-color: #fee2e2;
  color: #991b1b;
  border: 1px solid #fecaca;
}

.toast-warning {
  background-color: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}

.toast-success {
  background-color: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.sender-panel {
  transition: all 0.2s ease;
}

section.sender-panel.is-dragging {
  background-color: #f0f9ff;
  border: 2px dashed #0f6bd7;
  transform: scale(1.01);
}
</style>
