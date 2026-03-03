<script setup lang="ts">
import { onBeforeUnmount, ref, useAttrs, watch } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import sampleFileUrl from '../assets/sample-files/test.png';
import { useDemoRuntime } from '../demo-runtime';
import SpectrumCanvas from './SpectrumCanvas.vue';

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
const fileInput = ref<HTMLInputElement | null>(null);
const isTransmitting = ref(false);
const isPreparing = ref(false);
const senderStatus = ref('Idle');
const isDragging = ref(false);
const toasts = ref<Toast[]>([]);
let toastIdCounter = 0;

let senderWorker: Worker | null = null;
let senderBackend: Comlink.Remote<MistcastBackend> | null = null;
let encoderNode: AudioWorkletNode | null = null;
let analyserNode: AnalyserNode | null = null;

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

async function loadSampleFile(): Promise<Uint8Array> {
  const res = await fetch(sampleFileUrl);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

async function teardownSenderGraph() {
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

  encoderNode.connect(audioContext.destination);
  encoderNode.connect(analyserNode);
  encoderNode.connect(demoAirGapNode);
}

async function startSendingData(data: Uint8Array) {
  await runExclusive(async () => {
    if (!runtime.coreReady.value) return;
    isPreparing.value = true;
    try {
      const { audioContext } = await runtime.ensureAudioCore();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      senderStatus.value = 'Preparing...';
      await rebuildSenderGraph();
      if (!senderBackend) return;

      senderStatus.value = 'Transmitting...';
      isTransmitting.value = true;
      await senderBackend.startEncoder(data, audioContext.sampleRate, runtime.modemMode.value, runtime.randomizeSeq.value);
    } finally {
      isPreparing.value = false;
    }
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

watch(isTransmitting, (busy) => {
  runtime.isBusy.value = busy;
});

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
        <button @click="startSendingText" class="btn btn-primary" :disabled="!runtime.coreReady.value || isPreparing">
          {{ isPreparing ? 'Preparing...' : 'Send' }}
        </button>
        <button @click="startSendingSampleImage" class="btn" :disabled="!runtime.coreReady.value || isPreparing">Send Sample Image</button>
        <button @click="triggerFileSelect" class="btn" :disabled="!runtime.coreReady.value || isPreparing">Send File (max 4KB)</button>
        <input type="file" ref="fileInput" style="display: none" @change="handleFileSelect" />
      </template>
      <button v-else @click="stopSending" class="btn btn-danger" :disabled="!runtime.coreReady.value">Stop</button>
    </div>
    <SpectrumCanvas
      :analyser-node="analyserNode"
      title="Sender FFT (Linear Frequency Axis)"
    />
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
