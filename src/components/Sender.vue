<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, onUnmounted, ref, useAttrs, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import * as Comlink from 'comlink';
import type { MistcastBackend } from '../worker';
import MistcastWorker from '../worker?worker';
import { useDemoRuntime, type SenderStatus } from '../demo-runtime';
import SpectrumCanvas from './SpectrumCanvas.vue';

// Disable automatic attribute inheritance since we have multiple root nodes
defineOptions({
  inheritAttrs: false,
});

const runtime = useDemoRuntime();
const attrs = useAttrs();

const MAX_FILE_SIZE = 255 * 16; // 4080 bytes
const TOAST_DURATION_MS = 5000;
const STORAGE_KEY = 'sender-send-mode';

import sampleTestPng from '../assets/sample-files/test.png';
import sampleWebpWebp from '../assets/sample-files/webp.webp';

interface SampleFile {
  id: string;
  name: string;
  url: string;
  size: number;
}

const SAMPLE_FILES: SampleFile[] = [
  { id: 'png', name: 'test.png', url: sampleTestPng, size: 921 },
  { id: 'webp', name: 'webp.webp', url: sampleWebpWebp, size: 3874 },
];

const selectedSample = ref<SampleFile>(SAMPLE_FILES[0]);

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

type SendMode = 'text' | 'sample' | 'file';

const inputText = ref('Hello Acoustic World!');
const fileInput = ref<HTMLInputElement | null>(null);
const isTransmitting = ref(false);
const isPreparing = ref(false);
const senderStatus = ref<SenderStatus>('idle');

// runtimeにステータスを同期
watch(senderStatus, (val) => {
  runtime.senderStatus.value = val;
}, { immediate: true });

// runtimeに操作を登録
onMounted(() => {
  runtime.onStartSender.value = handleSend;
  runtime.onStopSender.value = stopSending;
});

onUnmounted(() => {
  runtime.onStartSender.value = null;
  runtime.onStopSender.value = null;
});

const isDragging = ref(false);
const toasts = ref<Toast[]>([]);
let toastIdCounter = 0;

const { t } = useI18n();

const displayStatus = computed(() => {
  switch (senderStatus.value) {
    case 'idle': return t('sender.status.idle');
    case 'ready': return t('sender.status.ready');
    case 'preparing': return t('sender.status.preparing');
    case 'transmitting': return t('sender.status.broadcasting');
    default: return senderStatus.value;
  }
});

// 送信モード管理
const sendMode = ref<SendMode>('text');
const selectedFile = ref<File | null>(null);

let senderWorker: Worker | null = null;
let senderBackend: Comlink.Remote<MistcastBackend> | null = null;
let encoderNode: AudioWorkletNode | null = null;
let analyserNode: AnalyserNode | null = null;

let opQueue: Promise<void> = Promise.resolve();

function validateFileSize(size: number): boolean {
  return size <= MAX_FILE_SIZE;
}

// localStorageから送信モードを復元
function loadSendMode(): SendMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'text' || stored === 'sample' || stored === 'file') {
      return stored;
    }
  } catch (e) {
    // localStorageアクセスエラーは無視
  }
  return 'text';
}

// 送信モードをlocalStorageに保存
function saveSendMode(mode: SendMode) {
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch (e) {
    // localStorageアクセスエラーは無視
  }
}

// 初期化: localStorageから送信モードを復元
sendMode.value = loadSendMode();

// 送信モードを監視してlocalStorageに保存
watch(sendMode, (newMode) => {
  saveSendMode(newMode);
}, { immediate: false });

function showToast(message: string, type: ToastType = 'error') {
  const id = toastIdCounter++;
  toasts.value.push({ id, message, type });
  setTimeout(() => {
    toasts.value = toasts.value.filter(t => t.id !== id);
  }, TOAST_DURATION_MS);
}

// Computed properties
const textByteSize = computed(() => {
  return new TextEncoder().encode(inputText.value).length;
});

const contentBytes = computed(() => {
  switch (sendMode.value) {
    case 'text':
      return textByteSize.value;
    case 'sample':
      return selectedSample.value.size;
    case 'file':
      return selectedFile.value?.size || 0;
  }
});

const sendButtonText = computed(() => {
  switch (sendMode.value) {
    case 'text':
      return inputText.value.trim() ? 'Send' : 'Send Text';
    case 'sample':
      return 'Send Sample Image';
    case 'file':
      return selectedFile.value ? `Send: ${selectedFile.value.name}` : 'Select File';
  }
});

const canSend = computed(() => {
  switch (sendMode.value) {
    case 'text':
      return inputText.value.trim().length > 0;
    case 'sample':
      return true;
    case 'file':
      return selectedFile.value !== null;
  }
});

// Handlers
function triggerFileSelect() {
  fileInput.value?.click();
}

function clearFile() {
  selectedFile.value = null;
  if (fileInput.value) {
    fileInput.value.value = '';
  }
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
    selectedFile.value = file;
    sendMode.value = 'file';
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
    selectedFile.value = file;
    sendMode.value = 'file';
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
  const res = await fetch(selectedSample.value.url);
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

      senderStatus.value = 'preparing';
      await rebuildSenderGraph();
      if (!senderBackend) return;

      senderStatus.value = 'transmitting';
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

// 統一送信ハンドラ
async function handleSend() {
  switch (sendMode.value) {
    case 'text':
      await startSendingText();
      break;
    case 'sample':
      await startSendingSampleImage();
      break;
    case 'file':
      if (selectedFile.value) {
        await startSendingFile(selectedFile.value);
      }
      break;
  }
}

async function stopSending() {
  await runExclusive(async () => {
    await senderBackend?.stopEncoder();
    isTransmitting.value = false;
    senderStatus.value = runtime.coreReady.value ? 'ready' : 'idle';
  });
}

watch(
  () => runtime.coreReady.value,
  async (ready) => {
    if (!ready) return;
    await runExclusive(async () => {
      await rebuildSenderGraph();
      senderStatus.value = 'ready';
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
    <div class="sender-header">
      <div class="sender-title-row">
        <h2>{{ $t('common.sender') }}</h2>
        <div class="status-chip" :class="senderStatus">
          {{ displayStatus }}
        </div>
      </div>
      <p class="panel-sub">{{ $t('settings.modem.label') }} / {{ $t('settings.transmission.randomize') }}</p>
    </div>

    <!-- 送信タイプ選択（セグメントコントロール風） -->
    <div class="send-type-tabs">
      <button
        type="button"
        class="tab-item"
        :class="{ active: sendMode === 'text' }"
        :disabled="isTransmitting"
        @click="sendMode = 'text'"
      >
        <span class="tab-icon">📝</span>
        <span class="tab-text">{{ $t('sender.mode.text') }}</span>
      </button>
      <button
        type="button"
        class="tab-item"
        :class="{ active: sendMode === 'sample' }"
        :disabled="isTransmitting"
        @click="sendMode = 'sample'"
      >
        <span class="tab-icon">🖼️</span>
        <span class="tab-text">{{ $t('sender.mode.sample') }}</span>
      </button>
      <button
        type="button"
        class="tab-item"
        :class="{ active: sendMode === 'file' }"
        :disabled="isTransmitting"
        @click="sendMode = 'file'"
      >
        <span class="tab-icon">📁</span>
        <span class="tab-text">{{ $t('sender.mode.file') }}</span>
      </button>
    </div>

    <!-- テキスト入力エリア（テキストモード時のみ表示） -->
    <div v-if="sendMode === 'text'" class="input-area">
      <textarea
        v-model="inputText"
        rows="4"
        :placeholder="$t('sender.input.placeholder')"
        :disabled="isTransmitting"
      />
    </div>

    <!-- サンプル画像選択 & プレビュー（サンプルモード時のみ表示） -->
    <div v-if="sendMode === 'sample'" class="sample-mode-container">
      <div class="sample-selector">
        <button
          v-for="sample in SAMPLE_FILES"
          :key="sample.id"
          class="sample-option-btn"
          :class="{ active: selectedSample.id === sample.id }"
          @click="selectedSample = sample"
        >
          {{ sample.name }} ({{ sample.size }}B)
        </button>
      </div>
      <div class="sample-preview">
        <img :src="selectedSample.url" :alt="selectedSample.name" />
        <div class="sample-info">{{ $t('sender.sample.info', { name: selectedSample.name, size: selectedSample.size }) }}</div>
      </div>
    </div>

    <!-- ファイル選択エリア（ファイルモード時のみ表示） -->
    <div v-if="sendMode === 'file'">
      <!-- ファイルプレビュー -->
      <div v-if="selectedFile" class="file-preview">
        <div class="file-icon">📄</div>
        <div class="file-details">
          <div class="file-name">{{ selectedFile.name }}</div>
          <div class="file-size">{{ selectedFile.size }} bytes</div>
        </div>
        <button class="remove-file-btn" @click="clearFile" :title="$t('common.clear')">×</button>
      </div>

      <!-- ドラッグ&ドロップエリア -->
      <div
        v-else
        class="drop-zone"
        :class="{ 'is-dragging': isDragging }"
        @click="triggerFileSelect"
        @dragenter.prevent="handleDragEnter"
        @dragover.prevent
        @dragleave.prevent="handleDragLeave"
        @drop.prevent="handleDrop"
      >
        <div class="drop-content">
          <div class="drop-text">{{ $t('sender.input.drop_hint') }}</div>
          <div class="drop-hint">MAX 4KB</div>
        </div>
      </div>
    </div>

    <!-- 隠しファイル入力 -->
    <input ref="fileInput" type="file" class="file-input-hidden" @change="handleFileSelect" />

    <!-- 統一送信ボタン -->
    <div class="send-footer">
      <div class="footer-meta">
        <div class="size-indicator" :class="{ warning: contentBytes > 3500 }">
          {{ contentBytes }} / 4080 bytes
        </div>
      </div>
      <template v-if="!isTransmitting">
        <button
          class="btn btn-primary btn-large"
          :disabled="!canSend || isPreparing || !runtime.coreReady.value"
          @click="handleSend"
        >
          {{ isPreparing ? $t('sender.status.preparing') : $t('common.start') }}
        </button>
      </template>
      <button v-else @click="stopSending" class="btn btn-danger btn-large">{{ $t('common.stop') }}</button>
    </div>

    <SpectrumCanvas
      :analyser-node="analyserNode"
      title="Sender FFT"
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

.sender-header {
  margin-bottom: 0.8rem;
  display: flex;
  flex-direction: column;
}

.sender-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.sender-title-row h2 {
  margin: 0;
}

.sender-title-row .status-chip {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

/* 送信タイプ選択（セグメントコントロール） */
.send-type-tabs {
  display: flex;
  background: #f1f5f9;
  padding: 4px;
  border-radius: 12px;
  margin-bottom: 1.25rem;
  border: 1px solid var(--line);
}

.tab-item {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.6rem 0.2rem;
  border: none;
  background: transparent;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--muted);
  font-weight: 600;
  font-size: 0.85rem;
  white-space: nowrap;
}

.tab-item:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.5);
  color: var(--ink);
}

.tab-item.active {
  background: #fff;
  color: var(--primary);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}

.tab-item:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tab-icon {
  font-size: 1.1rem;
}

/* 入力エリア */
.input-area {
  display: flex;
  flex-direction: column;
  margin-bottom: 1rem;
}

textarea:disabled {
  background-color: #f1f5f9;
  color: var(--muted);
  cursor: not-allowed;
  opacity: 0.8;
}

/* サンプルプレビュー */
.sample-mode-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.sample-selector {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.sample-option-btn {
  padding: 0.4rem 0.8rem;
  font-size: 0.8rem;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--muted);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.sample-option-btn:hover {
  background: #f1f5f9;
  color: var(--ink);
}

.sample-option-btn.active {
  background: var(--primary);
  color: #fff;
  border-color: var(--primary);
}

.sample-preview {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.2rem;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 12px;
}

.sample-preview img {
  width: 80px;
  height: 80px;
  object-fit: contain;
  background: #fff;
  border-radius: 8px;
  border: 1px solid var(--line);
  image-rendering: pixelated;
  padding: 4px;
}

.sample-info {
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--ink);
}

/* ファイルプレビュー */
.file-preview {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: #f6f9fc;
  border: 1px solid var(--line);
  border-radius: 10px;
  margin-bottom: 1rem;
}

.file-icon {
  font-size: 1.5rem;
}

.file-details {
  flex: 1;
  min-width: 0;
}

.file-name {
  font-weight: 600;
  color: var(--ink);
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-size {
  font-size: 0.75rem;
  color: var(--muted);
  font-family: var(--mono);
}

.remove-file-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.2s;
  font-size: 1.2rem;
  line-height: 1;
}

.remove-file-btn:hover {
  background: #fee2e2;
  color: #b94731;
}

/* ドロップゾーン */
.drop-zone {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100px;
  padding: 1.5rem;
  border: 2px dashed var(--line);
  border-radius: 10px;
  background: #fbfdff;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 1rem;
}

.drop-zone:hover {
  border-color: var(--primary);
  background: #f0f9ff;
}

.drop-zone.is-dragging {
  border-color: var(--primary);
  background: #f0f9ff;
  transform: scale(1.02);
}

.drop-content {
  text-align: center;
}

.drop-text {
  font-weight: 600;
  color: var(--ink);
  font-size: 0.95rem;
  margin-bottom: 0.25rem;
}

.drop-hint {
  font-size: 0.8rem;
  color: var(--muted);
}

/* 隠しファイル入力 */
.file-input-hidden {
  display: none;
}

/* 送信フッター */
.send-footer {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  margin-top: 1rem;
}

.footer-meta {
  display: flex;
  justify-content: flex-end;
}

.size-indicator {
  font-size: 0.8rem;
  color: var(--muted);
  font-family: var(--mono);
  white-space: nowrap;
}

.size-indicator.warning {
  color: #d97706;
  font-weight: 600;
}

/* モバイルファースト */
@media (max-width: 779px) {
  .sender-panel h2,
  .sender-panel .panel-sub {
    display: none;
  }

  .sender-header {
    margin-bottom: 0.5rem;
  }

  .sender-title-row {
    justify-content: flex-end;
  }

  .size-indicator {
    text-align: center;
  }
}
</style>
