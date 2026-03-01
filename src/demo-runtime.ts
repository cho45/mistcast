import { inject, provide, ref, type InjectionKey, type Ref } from 'vue';

export type AudioCore = {
  audioContext: AudioContext;
  demoAirGapNode: GainNode;
};

export type DemoRuntime = {
  coreReady: Ref<boolean>;
  ensureAudioCore: () => Promise<AudioCore>;
};

const DemoRuntimeKey: InjectionKey<DemoRuntime> = Symbol('DemoRuntime');

export function createDemoRuntime(ensureAudioCore: () => Promise<AudioCore>): DemoRuntime {
  const coreReady = ref(false);

  return {
    coreReady,
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
