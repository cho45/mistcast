<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue';
import Sender from './components/Sender.vue';
import Receiver from './components/Receiver.vue';
// @ts-ignore
import processorsUrl from './audio-processors?worker&url';
import { createDemoRuntime, provideDemoRuntime, type AudioCore } from './demo-runtime';

let audioContext: AudioContext | null = null;
let demoAirGapNode: GainNode | null = null;
let initPromise: Promise<AudioCore> | null = null;
const appStatus = ref('Idle');

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
    appStatus.value = 'Ready (Rx standby)';

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

const activeTab = ref<'sender' | 'receiver'>('receiver');

async function initialize() {
  if (runtime.coreReady.value) return;
  appStatus.value = 'Initializing...';
  try {
    await ensureAudioCore();
  } catch (e) {
    console.error(e);
    appStatus.value = 'Init Error';
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
</script>

<template>
  <div class="app-shell">
    <header class="hero">
      <h1>Mistcast Demo v2.3</h1>
      <p>Acoustic DSSS + RLNC playground</p>
      <div class="status-chip" :class="appStatus.toLowerCase().replace(/[^a-z0-9]+/g, '-')">
        {{ appStatus }}
      </div>
    </header>

    <main class="content">
      <section v-if="!runtime.coreReady.value" class="panel init-panel">
        <p>まず Audio System を初期化して、送受信ノードを作成します。</p>
        <button @click="initialize" class="btn btn-primary btn-large">Initialize Audio System</button>
      </section>

      <template v-else>
        <nav class="app-tabs">
          <button @click="activeTab = 'sender'" class="tab-btn" :class="{ active: activeTab === 'sender' }">Sender</button>
          <button @click="activeTab = 'receiver'" class="tab-btn" :class="{ active: activeTab === 'receiver' }">Receiver</button>
        </nav>
        <div class="panel-container">
          <Sender class="panel-item" :class="{ 'is-active': activeTab === 'sender' }" />
          <Receiver class="panel-item" :class="{ 'is-active': activeTab === 'receiver' }" />
        </div>
      </template>
    </main>

    <footer class="footnote">
      <div><strong>Demo Bridge:</strong> <code>[Sender] -demoAirGapNode- [Receiver]</code></div>
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

.app-tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.5rem;
}

.tab-btn {
  padding: 0.6rem 1.2rem;
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

.spectrum-panel {
  margin-top: 0.9rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #f8fbff;
  padding: 0.55rem;
}

.spectrum-title {
  margin: 0 0 0.45rem;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
}

.spectrum-canvas {
  display: block;
  width: 100%;
  height: 190px;
  border: 1px solid #d3e0ed;
  border-radius: 8px;
  background: #fdfefe;
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

  .proc-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
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
  max-width: 100%;
}
</style>
