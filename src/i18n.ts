import { createI18n } from 'vue-i18n';
import en from './locales/en.json';
import ja from './locales/ja.json';
import type { AppLanguage } from './composables/useSettings';

export function getBrowserLanguage(): 'en' | 'ja' {
  const lang = navigator.language.split('-')[0];
  return lang === 'ja' ? 'ja' : 'en';
}

export function resolveLanguage(lang: AppLanguage): 'en' | 'ja' {
  if (lang === 'auto') {
    return getBrowserLanguage();
  }
  return lang;
}

const i18n = createI18n({
  legacy: false,
  locale: 'en', // Default, will be updated on app mount
  fallbackLocale: 'en',
  messages: {
    en,
    ja
  }
});

export default i18n;
