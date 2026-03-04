<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';

interface Props {
  analyserNode: AnalyserNode | null;
  title?: string;
}

const props = withDefaults(defineProps<Props>(), {
  title: 'FFT (Linear Frequency Axis)'
});

const canvas = ref<HTMLCanvasElement | null>(null);
let fftData: Float32Array | null = null;
let rafId: number | null = null;

function drawFrame() {
  if (!props.analyserNode || !fftData || !canvas.value) {
    rafId = window.requestAnimationFrame(() => drawFrame());
    return;
  }

  const c = canvas.value;
  const ctx = c.getContext('2d');
  if (!ctx) {
    rafId = window.requestAnimationFrame(() => drawFrame());
    return;
  }

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(10, c.clientWidth || 640);
  const cssH = Math.max(10, c.clientHeight || 180);
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // @ts-expect-error - Float32Array<ArrayBufferLike> vs Float32Array<ArrayBuffer> type mismatch in TS versions
  props.analyserNode.getFloatFrequencyData(fftData);
  const nyquist = props.analyserNode.context.sampleRate / 2;
  const fMax = Math.max(1, Math.min(20000, nyquist));
  const minDb = props.analyserNode.minDecibels;
  const maxDb = props.analyserNode.maxDecibels;

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

  // Y軸のdBラベル
  for (const db of dbTicks) {
    const y = yFromDb(db);
    const yPos = Math.max(12, Math.min(cssH - 4, y + 3));
    ctx.fillText(`${db}`, 2, yPos);
  }

  rafId = window.requestAnimationFrame(() => drawFrame());
}

function start() {
  if (rafId !== null) return;
  if (!props.analyserNode) return;

  fftData = new Float32Array(props.analyserNode.frequencyBinCount);
  rafId = window.requestAnimationFrame(() => drawFrame());
}

function stop() {
  if (rafId !== null) {
    window.cancelAnimationFrame(rafId);
    rafId = null;
  }
}

watch(() => props.analyserNode, () => {
  if (props.analyserNode) {
    stop();
    start();
  } else {
    stop();
  }
});

onMounted(() => {
  if (props.analyserNode) {
    start();
  }
});

onBeforeUnmount(() => {
  stop();
});
</script>

<template>
  <div class="spectrum-panel">
    <p class="spectrum-title">{{ title }}</p>
    <canvas ref="canvas" class="spectrum-canvas"></canvas>
  </div>
</template>

<style scoped>
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
</style>
