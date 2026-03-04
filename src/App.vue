<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue';
import i18n, { resolveLanguage } from './i18n';
import Sender from './components/Sender.vue';
import Receiver from './components/Receiver.vue';
import Settings from './components/Settings.vue';
// @ts-ignore
import processorsUrl from './audio-processors?worker&url';
import { createDemoRuntime, provideDemoRuntime, type AudioCore } from './demo-runtime';
import { provideSettings } from './composables/useSettings';

let audioContext: AudioContext | null = null;
let demoAirGapNode: GainNode | null = null;
let initPromise: Promise<AudioCore> | null = null;
const isInitializing = ref(false);

async function ensureAudioCore(): Promise<AudioCore> {
  if (audioContext && demoAirGapNode) {
    return { audioContext, demoAirGapNode };
  }
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const context = new AudioContext({ sampleRate: 48000 });
    await context.audioWorklet.addModule(processorsUrl);
    const airGap = context.createGain();
    airGap.gain.value = 1.0;

    audioContext = context;
    demoAirGapNode = airGap;
    runtime.coreReady.value = true;

    return { audioContext: context, demoAirGapNode: airGap };
  })();

  try {
    return await initPromise;
  } finally {
    initPromise = null;
  }
}

const runtime = createDemoRuntime(ensureAudioCore);
provideDemoRuntime(runtime);

const { settings } = provideSettings();

// 言語設定の同期
watch(() => settings.value.language, (lang) => {
  (i18n.global.locale as any).value = resolveLanguage(lang);
}, { immediate: true });

const activeTab = ref<'sender' | 'receiver'>((localStorage.getItem('mistcast_active_tab') as any) || 'sender');
const settingsDialog = ref<HTMLDialogElement | null>(null);

watch(activeTab, (newTab) => {
  localStorage.setItem('mistcast_active_tab', newTab);
});

function openSettings() {
  settingsDialog.value?.showModal();
}

function closeSettings() {
  settingsDialog.value?.close();
}

function handleDialogClick(event: MouseEvent) {
  const rect = settingsDialog.value?.getBoundingClientRect();
  if (!rect) return;

  // dialog外をクリックした場合は閉じる
  const clickedInDialog =
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom;

  if (!clickedInDialog) {
    closeSettings();
  }
}

async function initialize() {
  if (runtime.coreReady.value || isInitializing.value) return;
  isInitializing.value = true;
  try {
    await ensureAudioCore();
  } catch (e) {
    console.error(e);
  } finally {
    isInitializing.value = false;
  }
}

onBeforeUnmount(() => {
  try {
    demoAirGapNode?.disconnect();
  } catch {
    // no-op
  }
  demoAirGapNode = null;

  if (audioContext && audioContext.state !== 'closed') {
    void audioContext.close();
  }
  audioContext = null;
});

defineExpose({
  activeTab,
  runtime,
  settingsDialog
});
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="app-header-content">
        <div class="app-brand-area">
          <div class="app-brand">
            <h1>Mistcast</h1>
          </div>
          <div class="header-status-area">
            <TransitionGroup name="header-chip">
              <!-- Sender Remote Toggle Chip -->
              <button
                v-if="runtime.coreReady.value"
                key="sender"
                class="header-chip sender-chip"
                :class="{ 'is-active': runtime.senderStatus.value === 'transmitting' }"
                @click.stop="runtime.senderStatus.value === 'transmitting' ? runtime.onStopSender.value?.() : runtime.onStartSender.value?.()"
              >
                <span class="chip-icon" :class="{ pulse: runtime.senderStatus.value === 'transmitting' }">📡</span>
                <span class="chip-action">{{ runtime.senderStatus.value === 'transmitting' ? $t('common.stop') : (runtime.senderStatus.value === 'preparing' ? '...' : $t('common.start')) }}</span>
              </button>

              <!-- Receiver Status Chip -->
              <button
                v-if="runtime.receiverProgress.value > 0 || runtime.receiverStatus.value === 'decoded'"
                key="receiver"
                class="header-chip receiver-chip"
                @click.stop="runtime.receiverStatus.value === 'decoded' ? runtime.onResetReceiver.value?.() : (activeTab = 'receiver')"
              >
                <span class="chip-icon" v-if="runtime.receiverStatus.value !== 'decoded'">📥</span>
                <span class="chip-icon" v-else>✅</span>
                <span class="chip-label" v-if="runtime.receiverStatus.value !== 'decoded'">
                  {{ `${(runtime.receiverProgress.value * 100).toFixed(0)}%` }}
                </span>
                <span v-if="runtime.receiverStatus.value === 'decoded'" class="chip-action">{{ $t('common.reset') }}</span>
              </button>
            </TransitionGroup>
          </div>
        </div>
        <button class="settings-trigger" @click="openSettings" :aria-label="$t('common.settings')">
          <img src="./assets/settings.svg" :alt="$t('common.settings')" class="settings-icon" />
        </button>
      </div>
    </header>

    <dialog ref="settingsDialog" class="settings-dialog" @click="handleDialogClick">
      <button class="dialog-close" @click="closeSettings" :aria-label="$t('common.close')">×</button>
      <div class="dialog-content">
        <Settings />
      </div>
    </dialog>

    <main class="content">
      <section v-if="!runtime.coreReady.value" class="panel init-panel">
        <div class="init-content">
          <p class="init-desc">{{ $t('common.init_desc') }}</p>
          <ul class="init-features">
            <li>{{ $t('common.init_features.acoustic') }}</li>
            <li>{{ $t('common.init_features.airgap') }}</li>
            <li>{{ $t('common.init_features.browser') }}</li>
          </ul>
        </div>

        <div class="init-action">
          <p class="init-hint">{{ $t('common.init_info') }}</p>
          <button
            @click="initialize"
            class="btn btn-primary btn-large btn-init"
            :disabled="isInitializing"
          >
            {{ isInitializing ? $t('sender.status.preparing') : $t('common.start') }}
          </button>
        </div>
      </section>

      <template v-else>
        <nav class="app-tabs">
          <button @click="activeTab = 'sender'" class="tab-btn" :class="{ active: activeTab === 'sender' }">{{ $t('common.sender') }}</button>
          <button @click="activeTab = 'receiver'" class="tab-btn" :class="{ active: activeTab === 'receiver' }">{{ $t('common.receiver') }}</button>
        </nav>
        <div class="panel-container">
          <Sender class="panel-item" :class="{ 'is-active': activeTab === 'sender' }" />
          <Receiver class="panel-item" :class="{ 'is-active': activeTab === 'receiver' }" />
        </div>
      </template>
    </main>

    <footer class="footnote">
      <a href="https://github.com/cho45/mistcast" target="_blank" rel="noopener noreferrer">GitHub</a>
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
  padding: 0;
  color: var(--ink);
  overflow-x: hidden;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  font-family: var(--sans);
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 100%;
}

.app-header {
  background: linear-gradient(to bottom, #f8fafc, #f1f5f9);
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
}

.app-header-content {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0.6rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-height: 3.5rem;
}

.app-brand-area {
  display: flex;
  align-items: center;
  gap: 1rem;
  min-width: 0;
}

.header-status-area {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-chip {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.3rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #fff;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
  font-size: 0.8rem;
  font-weight: 700;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  min-height: 2rem;
}

.header-chip:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
  border-color: var(--primary);
}

.header-chip:active {
  transform: translateY(0);
}

.sender-chip {
  background: #f8fafc;
  border-color: #cbd5e1;
  color: #64748b;
}

.sender-chip.is-active {
  background: #fff4d8;
  border-color: #f0d299;
  color: #8e5a00;
}

.receiver-chip {
  background: #e4f2fb;
  border-color: #a9cfe6;
  color: #0a557f;
}

.chip-icon.pulse {
  animation: headerPulse 1.5s infinite ease-in-out;
}

@keyframes headerPulse {
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
  100% { transform: scale(1); opacity: 1; }
}

.chip-action {
  padding: 0.1rem 0.4rem;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 6px;
  font-size: 0.7rem;
  text-transform: uppercase;
  color: var(--muted);
}

.is-active .chip-action {
  background: rgba(142, 90, 0, 0.1);
  color: #8e5a00;
}

.receiver-chip .chip-action {
  background: rgba(10, 85, 127, 0.1);
  color: #0a557f;
}

.header-chip:hover .chip-action {
  background: var(--primary);
  color: #fff;
}

.sender-chip.is-active:hover .chip-action { background: var(--danger); }

/* Animations */
.header-chip-enter-active,
.header-chip-leave-active {
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.header-chip-enter-from {
  opacity: 0;
  transform: scale(0.8) translateY(-10px);
}

.header-chip-leave-to {
  opacity: 0;
  transform: scale(0.8) translateY(10px);
}

.app-brand h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  letter-spacing: -0.01em;
}

@media (max-width: 480px) {
  .app-brand h1 {
    font-size: 1rem;
  }
  
  .app-brand-area {
    gap: 0.5rem;
  }

  .header-chip {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
  }
}

.settings-trigger {
  width: 2.25rem;
  height: 2.25rem;
  border: none;
  background: transparent;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border-radius: 6px;
}

.settings-trigger:hover {
  background: rgba(0, 0, 0, 0.05);
}

.settings-trigger:active {
  background: rgba(0, 0, 0, 0.08);
}

.settings-icon {
  width: 18px;
  height: 18px;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.settings-trigger:hover .settings-icon {
  opacity: 0.9;
}

.settings-trigger:hover .settings-icon {
  opacity: 1;
}

.settings-dialog {
  border: none;
  border-radius: 20px;
  padding: 0;
  max-width: 640px;
  width: 90vw;
  max-height: 85vh;
  box-shadow: 0 25px 80px rgba(0, 0, 0, 0.35);
  animation: dialogFadeIn 0.3s ease-out;
}

@keyframes dialogFadeIn {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(-10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.settings-dialog::backdrop {
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  animation: backdropFadeIn 0.3s ease-out;
}

@keyframes backdropFadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.dialog-content {
  max-height: 85vh;
  overflow-y: auto;
  background: #fff;
  border-radius: 20px;
}

.dialog-close {
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 2.25rem;
  height: 2.25rem;
  border: none;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 50%;
  font-size: 1.5rem;
  cursor: pointer;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  color: var(--muted);
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.dialog-close:hover {
  background: #fff;
  color: var(--ink);
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.dialog-close:active {
  transform: scale(0.95);
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
.status-chip.ready,
.status-chip.mic-active-rx,
.status-chip.internal-loopback {
  color: #0a557f;
  background: #e4f2fb;
  border-color: #a9cfe6;
}

.status-chip.preparing {
  color: #0f6bd7;
  background: #f0f7ff;
  border-color: #cce3ff;
}

.status-chip.mic-error {
  color: #7e2d1f;
  background: #fdebe7;
  border-color: #d8b2a9;
}

.status-chip.idle {
  color: #5a6470;
  background: #f1f5f9;
  border-color: #d8dee8;
}

.content {
  margin: 0 auto;
  padding: 1.5rem 1rem;
  flex: 1;
  width: 100%;
  max-width: 1100px;
}

.app-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.5rem;
}

.tab-btn {
  padding: 0.8rem 1rem;
  border-radius: 8px;
  border: 1px solid transparent;
  background: transparent;
  cursor: pointer;
  font-weight: 500;
  color: var(--muted);
  transition: all 0.2s;
}

.tab-btn:hover {
  background: #f1f5f9;
}

.tab-btn.active {
  background: var(--primary);
  color: #fff;
}

.panel-container {
  display: block;
}

.panel-item {
  display: none;
}

.panel-item.is-active {
  display: block;
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

.receiver-title-row h2 {
  margin: 0;
}

.panel-sub {
  margin: 0.2rem 0 0.9rem;
  color: var(--muted);
  font-size: 0.84rem;
}

.init-panel {
  max-width: 560px;
  margin: 3rem auto;
  padding: 3rem 2rem;
  text-align: center;
}

.init-content {
  margin-bottom: 2.5rem;
}

.init-desc {
  font-size: 1.1rem;
  font-weight: 500;
  color: #1e293b;
  line-height: 1.6;
  margin-bottom: 1.5rem;
}

.init-features {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  text-align: center;
}

.init-features li {
  font-size: 0.95rem;
  color: var(--muted);
}

.init-features li::before {
  content: '•';
  margin-right: 0.5rem;
  color: var(--primary);
  font-weight: bold;
}

.init-action {
  padding-top: 2rem;
  border-top: 1px solid #e2e8f0;
}

.init-hint {
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 1.25rem;
}

.btn-init {
  padding: 1rem 3rem;
  font-size: 1.1rem;
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

.button-row-2col {
  display: flex;
  gap: 0.5rem;
}

.button-row-2col .btn {
  flex: 1;
  min-width: 0;
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

.receiver-title-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.receiver-title-row .status-chip {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
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

.metric small {
  display: block;
  font-size: 0.65em;
  color: var(--muted);
  margin-top: 0.15rem;
  font-weight: 400;
}

.proc-stats {
  margin-top: 0.8rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #f8fbff;
  padding: 0.6rem;
  min-width: 0;
}

.proc-title {
  margin: 0 0 0.4rem;
  font-size: 0.78rem;
  color: var(--muted);
  font-weight: 700;
}

.proc-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.35rem 0.5rem;
}

.proc-grid div {
  min-width: 0;
}

.proc-grid span {
  display: block;
  color: var(--muted);
  font-size: 0.7rem;
}

.proc-grid strong {
  font-family: var(--mono);
  font-size: 0.78rem;
  overflow-wrap: anywhere;
}

.proc-grid small {
  display: block;
  font-size: 0.65em;
  color: var(--muted);
  margin-top: 0.15rem;
  font-weight: 400;
}

.rx-log {
  margin-top: 0.85rem;
  max-height: 220px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfcfe;
}

.rx-log-header {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.35rem 0.45rem;
  border-bottom: 1px solid var(--line);
  background: #f2f7fc;
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 700;
}

.btn-xs {
  padding: 0.18rem 0.45rem;
  border-radius: 7px;
  font-size: 0.72rem;
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
  image-rendering: pixelated;
}

.image-meta {
  margin: 0;
  color: var(--muted);
  font-size: 0.8rem;
  font-family: var(--mono);
}

.footnote {
  margin: 0 auto 1rem;
  padding: 0 1rem 1rem;
  color: var(--muted);
  font-size: 0.84rem;
  text-align: center;
  width: 100%;
  max-width: 1100px;
}

.footnote a {
  color: var(--muted);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: all 0.2s;
}

.footnote a:hover {
  color: var(--primary);
  border-bottom-color: var(--primary);
}

@media (min-width: 780px) {
  .app-tabs {
    display: none;
  }

  .panel-container {
    display: grid;
    gap: 1rem;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    min-width: 0;
  }

  .panel-item {
    display: block !important;
  }

  .button-row {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .button-row.compact {
    grid-template-columns: repeat(2, minmax(0, max-content));
  }

  .button-row-2col {
    display: flex;
    gap: 0.5rem;
  }

  .button-row-2col .btn {
    flex: 1;
  }

  .metric-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .proc-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .settings-dialog {
    max-width: 700px;
  }
}

.basis-panel {
  margin-top: 1rem;
  padding: 0.8rem;
  background: #f1f5f9;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.6rem;
}

.basis-title {
  margin: 0;
  font-size: 0.8rem;
  color: #64748b;
  font-family: var(--mono);
}

.basis-canvas {
  background: #fff;
  border: 1px solid #cbd5e1;
  image-rendering: pixelated;
  max-width: 50%;
  aspect-ratio: 1 / 1;
  height: auto;
}

/* Tooltip styles */
.metric[data-tooltip],
.proc-grid div[data-tooltip],
.basis-panel[data-tooltip] {
  position: relative;
  cursor: help;
}

.metric[data-tooltip]::before,
.metric[data-tooltip]::after,
.proc-grid div[data-tooltip]::before,
.proc-grid div[data-tooltip]::after,
.basis-panel[data-tooltip]::before,
.basis-panel[data-tooltip]::after {
  position: absolute;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease;
  z-index: 100;
}

.metric[data-tooltip]::before,
.proc-grid div[data-tooltip]::before,
.basis-panel[data-tooltip]::before {
  content: attr(data-tooltip);
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%) translateY(-4px);
  background: #1b2229;
  color: #fff;
  padding: 0.5em 0.8em;
  border-radius: 6px;
  font-size: 0.85em;
  font-weight: 400;
  width: 16em;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.metric[data-tooltip]::after,
.proc-grid div[data-tooltip]::after,
.basis-panel[data-tooltip]::after {
  content: '';
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%) translateY(4px);
  border: 5px solid transparent;
  border-top-color: #1b2229;
}

.metric[data-tooltip]:hover::before,
.metric[data-tooltip]:hover::after,
.proc-grid div[data-tooltip]:hover::before,
.proc-grid div[data-tooltip]:hover::after,
.basis-panel[data-tooltip]:hover::before,
.basis-panel[data-tooltip]:hover::after {
  opacity: 1;
}
</style>
