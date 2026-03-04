import { ref } from 'vue';
import { mount } from '@vue/test-utils';
import { DemoRuntimeKey, type ModemMode, type SenderStatus, type ReceiverStatus } from '../demo-runtime';
import { SettingsKey, type AppSettings } from '../composables/useSettings';
import i18n from '../i18n';

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

export function createMockSettings() {
  const settings = ref<AppSettings>({
    modemMode: 'mary',
    debugMode: false,
    randomizeSeq: false,
    language: 'auto'
  });
  return {
    settings,
    saveSettings: () => {}
  };
}

/**
 * マウント対象コンポーネントにruntimeとsettingsを提供し、
 * i18nプラグインをインストールしてマウントする。
 */
export function mountApp(component: any, options: any = {}) {
  return mount(component, {
    ...options,
    global: {
      ...options.global,
      plugins: [i18n, ...(options.global?.plugins || [])],
      provide: {
        [DemoRuntimeKey as any]: createMockRuntime(),
        [SettingsKey as any]: createMockSettings(),
        ...options.global?.provide
      }
    }
  });
}
