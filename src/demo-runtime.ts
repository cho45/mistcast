import { inject, provide, ref, type InjectionKey, type Ref } from 'vue';

export type ModemMode = 'dsss' | 'mary';

export type AudioCore = {
  audioContext: AudioContext;
  demoAirGapNode: GainNode;
};

export type SenderStatus = 'idle' | 'ready' | 'preparing' | 'transmitting';
export type ReceiverStatus = 'idle' | 'receiving' | 'decoded' | 'error' | 'ready-rx-standby' | 'mic-active-rx' | 'internal-loopback' | 'mic-error';

export type DemoRuntime = {
  coreReady: Ref<boolean>;
  modemMode: Ref<ModemMode>;
  randomizeSeq: Ref<boolean>;
  isBusy: Ref<boolean>;
  senderStatus: Ref<SenderStatus>;
  receiverStatus: Ref<ReceiverStatus>;
  receiverProgress: Ref<number>;
  onStartSender: Ref<(() => void) | null>;
  onStopSender: Ref<(() => void) | null>;
  onResetReceiver: Ref<(() => void) | null>;
  ensureAudioCore: () => Promise<AudioCore>;
};

export const DemoRuntimeKey: InjectionKey<DemoRuntime> = Symbol('DemoRuntime');

export function createDemoRuntime(ensureAudioCore: () => Promise<AudioCore>): DemoRuntime {
  const coreReady = ref(false);
  const modemMode = ref<ModemMode>('mary');
  const randomizeSeq = ref(false);
  const isBusy = ref(false);
  const senderStatus = ref<SenderStatus>('idle');
  const receiverStatus = ref<ReceiverStatus>('idle');
  const receiverProgress = ref(0);
  const onStartSender = ref<(() => void) | null>(null);
  const onStopSender = ref<(() => void) | null>(null);
  const onResetReceiver = ref<(() => void) | null>(null);

  return {
    coreReady,
    modemMode,
    randomizeSeq,
    isBusy,
    senderStatus,
    receiverStatus,
    receiverProgress,
    onStartSender,
    onStopSender,
    onResetReceiver,
    ensureAudioCore,
  };
}

export function provideDemoRuntime(runtime: DemoRuntime) {
  provide(DemoRuntimeKey, runtime);
}

export function useDemoRuntime(): DemoRuntime {
  const runtime = inject(DemoRuntimeKey);
  if (!runtime) {
    throw new Error('DemoRuntime is not provided');
  }
  return runtime;
}
