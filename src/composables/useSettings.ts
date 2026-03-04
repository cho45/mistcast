import { inject, provide, ref, watch, type InjectionKey } from 'vue';

export type AppLanguage = 'auto' | 'en' | 'ja';

export interface AppSettings {
  modemMode: 'dsss' | 'mary';
  debugMode: boolean;
  randomizeSeq: boolean;
  language: AppLanguage;
}

const STORAGE_KEY = 'mistcast_settings';

const DEFAULT_SETTINGS: AppSettings = {
  modemMode: 'mary',
  debugMode: false,
  randomizeSeq: false,
  language: 'auto',
};

export const SettingsKey: InjectionKey<ReturnType<typeof useSettings>> = Symbol('Settings');

function loadSettings(): AppSettings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
  return { ...DEFAULT_SETTINGS };
}

export function useSettings() {
  const settings = ref<AppSettings>(loadSettings());

  function saveSettings() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings.value));
    } catch (e) {
      console.error('Failed to save settings:', e);
    }
  }

  // 設定変更時に自動保存
  watch(settings, saveSettings, { deep: true });

  return {
    settings,
    saveSettings,
  };
}

export function provideSettings() {
  const settings = useSettings();
  provide(SettingsKey, settings);
  return settings;
}

export function injectSettings() {
  const settings = inject(SettingsKey);
  if (!settings) {
    throw new Error('Settings is not provided');
  }
  return settings;
}
