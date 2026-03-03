import { inject, provide, ref, type InjectionKey, type Ref } from 'vue';

export type ModemMode = 'dsss' | 'mary';

export type AudioCore = {
  audioContext: AudioContext;
  demoAirGapNode: GainNode;
};

export type DemoRuntime = {
  coreReady: Ref<boolean>;
  modemMode: Ref<ModemMode>;
  randomizeSeq: Ref<boolean>;
  isBusy: Ref<boolean>;
  ensureAudioCore: () => Promise<AudioCore>;
};

const DemoRuntimeKey: InjectionKey<DemoRuntime> = Symbol('DemoRuntime');

export function createDemoRuntime(ensureAudioCore: () => Promise<AudioCore>): DemoRuntime {
  const coreReady = ref(false);
  const modemMode = ref<ModemMode>('mary');
  const randomizeSeq = ref(false);
  const isBusy = ref(false);

  return {
    coreReady,
    modemMode,
    randomizeSeq,
    isBusy,
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
