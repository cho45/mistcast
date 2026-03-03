import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useSettings, type AppSettings } from './useSettings';

// localStorage をモック
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    clear: () => {
      store = {};
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();

Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

describe('useSettings', () => {
  beforeEach(() => {
    // localStorage をクリア
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('初期値はデフォルト設定', () => {
    const { settings } = useSettings();
    expect(settings.value.modemMode).toBe('mary');
    expect(settings.value.debugMode).toBe(false);
    expect(settings.value.randomizeSeq).toBe(false);
  });

  it('localStorage から設定を読み込める', () => {
    localStorage.setItem('mistcast_settings', JSON.stringify({
      modemMode: 'dsss',
      debugMode: true,
      randomizeSeq: true,
    }));

    const { settings } = useSettings();
    expect(settings.value.modemMode).toBe('dsss');
    expect(settings.value.debugMode).toBe(true);
    expect(settings.value.randomizeSeq).toBe(true);
  });

  it('設定変更時に localStorage に保存される', async () => {
    const { settings } = useSettings();

    settings.value.modemMode = 'dsss';

    // watch は非同期なので次のティックを待つ
    await new Promise(resolve => setTimeout(resolve, 0));

    const stored = localStorage.getItem('mistcast_settings');
    expect(stored).toBeTruthy();
    const parsed = JSON.parse(stored!);
    expect(parsed.modemMode).toBe('dsss');
  });

  it('localStorage の破損データはデフォルト値でフォールバック', () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    localStorage.setItem('mistcast_settings', 'invalid json');

    const { settings } = useSettings();
    expect(settings.value.modemMode).toBe('mary');
    expect(settings.value.debugMode).toBe(false);
    expect(settings.value.randomizeSeq).toBe(false);

    consoleErrorSpy.mockRestore();
  });

  it('部分的な設定はデフォルト値とマージされる', () => {
    localStorage.setItem('mistcast_settings', JSON.stringify({
      modemMode: 'dsss',
    }));

    const { settings } = useSettings();
    expect(settings.value.modemMode).toBe('dsss');
    expect(settings.value.debugMode).toBe(false);
    expect(settings.value.randomizeSeq).toBe(false);
  });

  it('saveSettings を手動で呼び出せる', () => {
    const { settings, saveSettings } = useSettings();

    settings.value.modemMode = 'dsss';
    saveSettings();

    const stored = localStorage.getItem('mistcast_settings');
    expect(stored).toBeTruthy();
    const parsed = JSON.parse(stored!);
    expect(parsed.modemMode).toBe('dsss');
  });
});
