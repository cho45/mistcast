import { vi } from 'vitest';

const hasOwn = (obj: object, key: PropertyKey) =>
  Object.prototype.hasOwnProperty.call(obj, key);

if (!hasOwn(globalThis, "sampleRate")) {
  Object.defineProperty(globalThis, "sampleRate", {
    configurable: true,
    value: 48_000,
    writable: true,
  });
}

if (!hasOwn(globalThis, "currentTime")) {
  Object.defineProperty(globalThis, "currentTime", {
    configurable: true,
    value: 0,
    writable: true,
  });
}

// Mock AudioContext for component tests
class MockAudioContext {
  sampleRate = 48000;
  state: AudioContextState = 'running';
  currentTime = 0;
  audioWorklet = {
    addModule: vi.fn(() => Promise.resolve()),
  };

  async resume() {
    this.state = 'running';
    return this;
  }

  async close() {
    this.state = 'closed';
  }

  createAnalyser() {
    return {
      fftSize: 4096,
      smoothingTimeConstant: 0.6,
      minDecibels: -100,
      maxDecibels: -20,
      frequencyBinCount: 2048,
      getFloatFrequencyData: () => {},
    };
  }

  createGain() {
    return {
      gain: { value: 1.0 },
      connect: () => {},
      disconnect: () => {},
    };
  }

  async suspend() {
    this.state = 'suspended';
  }

  createMediaStreamDestination() {
    return {};
  }
}

// Mock Worker for component tests
class MockWorker {
  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  constructor(url: string | URL) {
    this.url = typeof url === 'string' ? url : url.href;
  }

  postMessage(data: unknown, transfer?: Transferable[]) {
    // Workerで処理されるべきメッセージをモック
    // 実際の処理は行わないが、エラーを投げないようにする
    vi.fn()();
  }

  terminate() {
    // Workerの終了処理をモック
    vi.fn();
  }

  addEventListener(type: string, listener: EventListener) {
    if (type === 'message') {
      this.onmessage = listener as ((event: MessageEvent) => void);
    } else if (type === 'error') {
      this.onerror = listener as ((event: ErrorEvent) => void);
    }
  }

  removeEventListener(type: string, listener: EventListener) {
    if (type === 'message') {
      this.onmessage = null;
    } else if (type === 'error') {
      this.onerror = null;
    }
  }
}

// Add AudioContext to global scope
if (!hasOwn(globalThis, "AudioContext")) {
  Object.defineProperty(globalThis, "AudioContext", {
    configurable: true,
    value: MockAudioContext,
    writable: true,
  });
}

// Add Worker to global scope
if (!hasOwn(globalThis, "Worker")) {
  Object.defineProperty(globalThis, "Worker", {
    configurable: true,
    value: MockWorker,
    writable: true,
  });
}

// Mock the audio-processors worker URL
if (!hasOwn(globalThis, "__mock_worker_url__")) {
  Object.defineProperty(globalThis, "__mock_worker_url__", {
    configurable: true,
    value: true,
    writable: true,
  });
}
