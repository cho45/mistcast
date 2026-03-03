import { h, ref } from 'vue';
import { provideDemoRuntime, type ModemMode } from '../demo-runtime';

export function createMockRuntime() {
  return {
    coreReady: ref(false),
    modemMode: ref<ModemMode>('mary'),
    randomizeSeq: ref(false),
    isBusy: ref(false),
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
      return () => h(component);
    },
  };
  return TestWrapper;
}
