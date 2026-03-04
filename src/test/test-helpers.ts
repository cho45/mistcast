import { h, ref } from 'vue';
import { provideDemoRuntime, type ModemMode, type SenderStatus, type ReceiverStatus } from '../demo-runtime';
import { provideSettings } from '../composables/useSettings';

export function createMockRuntime() {
  return {
    coreReady: ref(false),
    modemMode: ref<ModemMode>('mary'),
    randomizeSeq: ref(false),
    isBusy: ref(false),
    senderStatus: ref<SenderStatus>('idle'),
    receiverStatus: ref<ReceiverStatus>('idle'),
    receiverProgress: ref(0),
    onStartSender: ref<(() => void) | null>(null),
    onStopSender: ref<(() => void) | null>(null),
    onResetReceiver: ref<(() => void) | null>(null),
    ensureAudioCore: async () => ({
      audioContext: {
        sampleRate: 48000,
        state: 'running',
        resume: async () => {},
        createAnalyser: () => ({
          fftSize: 4096,
          smoothingTimeConstant: 0.6,
          minDecibels: -100,
          maxDecibels: -20,
          frequencyBinCount: 2048,
          getFloatFrequencyData: () => {},
        }),
        createGain: () => ({
          connect: () => {},
          disconnect: () => {},
        }),
        createMediaStreamDestination: () => ({}),
      } as any,
      demoAirGapNode: {
        connect: () => {},
        disconnect: () => {},
      } as any,
    }),
  };
}

export function mountWithRuntime(component: any) {
  const TestWrapper = {
    setup() {
      provideDemoRuntime(createMockRuntime());
      provideSettings();
      return () => h(component);
    },
  };
  return TestWrapper;
}
