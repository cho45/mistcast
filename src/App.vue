<script setup lang="ts">
import { ref, onMounted } from 'vue';
import * as Comlink from 'comlink';
import type { MistcastBackend } from './worker';

// Vite's worker loading
import MistcastWorker from './worker?worker';
// AudioWorklet module URL
import processorsUrl from './audio-processors?url';

const inputText = ref("Hello Acoustic World!");
const outputText = ref("");
const status = ref("Idle");
const isMicActive = ref(false);

// 進捗情報
const receivedPackets = ref(0);
const totalNeededPackets = ref(0);
const progressPercent = ref(0);

let backend: Comlink.Remote<MistcastBackend> | null = null;
let audioContext: AudioContext | null = null;
let encoderNode: AudioWorkletNode | null = null;
let decoderNode: AudioWorkletNode | null = null;
let micSource: MediaStreamAudioSourceNode | null = null;

async function init() {
  if (backend) return;
  status.value = "Initializing...";

  const worker = new MistcastWorker();
  backend = Comlink.wrap<MistcastBackend>(worker);
  await backend.init();

  audioContext = new AudioContext({ sampleRate: 48000 });
  await audioContext.audioWorklet.addModule(processorsUrl);

  // 1. Encoder Node (Speaker用)
  encoderNode = new AudioWorkletNode(audioContext, 'encoder-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [2]
  });
  const encChannel = new MessageChannel();
  encoderNode.port.postMessage({ type: 'attach-input-port', port: encChannel.port1 }, [encChannel.port1]);
  await backend.setAudioOutPort(Comlink.transfer(encChannel.port2, [encChannel.port2]));

  // 2. Decoder Node (Capture用)
  decoderNode = new AudioWorkletNode(audioContext, 'decoder-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 0,
  });
  const decChannel = new MessageChannel();
  decoderNode.port.postMessage({ type: 'attach-output-port', port: decChannel.port1 }, [decChannel.port1]);
  await backend.setAudioInPort(Comlink.transfer(decChannel.port2, [decChannel.port2]));

  // 接続: Encoder は常にスピーカーへ
  encoderNode.connect(audioContext.destination);
  
  // デフォルト接続: Encoder -> Decoder (内部ループバック)
  encoderNode.connect(decoderNode);
  
  status.value = "Ready";
}

async function startSending() {
  if (!backend || !audioContext) return;
  if (audioContext.state === 'suspended') await audioContext.resume();

  status.value = "Transmitting...";
  outputText.value = "";
  receivedPackets.value = 0;
  progressPercent.value = 0;

  const data = new TextEncoder().encode(inputText.value);
  
  // デコーダを準備 (K=8)
  await backend.startDecoder(
    data.length, 
    8, 
    audioContext.sampleRate, 
    Comlink.proxy((recovered: Uint8Array) => {
        outputText.value = new TextDecoder().decode(recovered);
        status.value = "Decoded!";
    }),
    Comlink.proxy((p: any) => {
        receivedPackets.value = p.received;
        totalNeededPackets.value = p.needed;
        progressPercent.value = p.progress;
    })
  );

  // 無限ストリーム送信開始
  await backend.startEncoder(data, audioContext.sampleRate);
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
      // 2重音（リーク）を防ぐため echoCancellation 等を無効化し純粋な入力を得る
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
        } 
      });
      micSource = audioContext.createMediaStreamSource(stream);
      
      // 内部ループバックを切断し、マイクをデコーダに繋ぐ
      encoderNode.disconnect(decoderNode);
      micSource.connect(decoderNode);
      
      isMicActive.value = true;
      status.value = "Mic Active";
    } catch (e) {
      console.error(e);
      status.value = "Mic Error";
    }
  } else {
    // マイクを切断し、内部ループバックに戻す
    micSource?.disconnect();
    micSource = null;
    encoderNode.connect(decoderNode);
    isMicActive.value = false;
    status.value = "Internal Loopback";
  }
}

async function reset() {
    await backend?.resetDecoder();
    outputText.value = "";
    receivedPackets.value = 0;
    progressPercent.value = 0;
    status.value = "Ready";
}
</script>

<template>
  <div class="container">
    <h1>Mistcast Demo v2.1</h1>
    
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
            <button @click="startSending" class="btn btn-primary" :disabled="status === 'Transmitting...'">Start Transmission</button>
            <button @click="stopSending" class="btn">Stop</button>
          </div>
        </div>

        <div class="recv-section">
          <h3>Receiver</h3>
          <div class="btn-group">
            <button @click="toggleMic" :class="{ 'btn-active': isMicActive }" class="btn">
              {{ isMicActive ? 'Disable Mic' : 'Enable Microphone' }}
            </button>
            <button @click="reset" class="btn">Clear</button>
          </div>
          
          <div class="progress-container" v-if="totalNeededPackets > 0">
            <div class="progress-text">
              Packets: {{ receivedPackets }} / {{ totalNeededPackets }} ({{ (progressPercent * 100).toFixed(1) }}%)
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" :style="{ width: (progressPercent * 100) + '%' }"></div>
            </div>
          </div>

          <div class="display">
            <p><strong>Decoded Result:</strong></p>
            <pre>{{ outputText }}</pre>
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
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}
.card {
  border: 1px solid #ddd;
  padding: 2rem;
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.status-bar {
  padding: 0.5rem 1rem;
  border-radius: 4px;
  margin-bottom: 1.5rem;
  font-weight: bold;
  background: #eee;
  text-align: center;
  transition: all 0.3s;
}
.status-bar.transmitting { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
.status-bar.decoded { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.status-bar.ready { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
.status-bar.mic-active { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

.controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
}
textarea {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin-bottom: 1rem;
  font-size: 1rem;
}
.btn-group {
  display: flex;
  gap: 0.5rem;
}

.progress-container {
  margin-top: 1.5rem;
}
.progress-text {
  font-size: 0.85rem;
  margin-bottom: 0.4rem;
  color: #666;
}
.progress-bar-bg {
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  background: #28a745;
  transition: width 0.2s ease-out;
}

.display {
  margin-top: 1rem;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  padding: 1rem;
  border-radius: 4px;
  min-height: 120px;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 1.1rem;
  color: #007bff;
  margin: 0;
}
button {
  padding: 0.6rem 1.2rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}
button:hover { background: #f0f0f0; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: #007bff; color: #fff; border-color: #0056b3; }
.btn-primary:hover:not(:disabled) { background: #0069d9; }
.btn-active { background: #dc3545; color: #fff; border-color: #bd2130; }
.btn-large { width: 100%; font-size: 1.2rem; padding: 1rem; }

.info {
  margin-top: 2rem;
  font-size: 0.9rem;
  color: #666;
  background: #f1f1f1;
  padding: 1rem;
  border-radius: 8px;
}
code { background: #e0e0e0; padding: 0.2rem 0.4rem; border-radius: 3px; }
</style>
