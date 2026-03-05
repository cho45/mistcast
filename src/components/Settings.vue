<script setup lang="ts">
import { injectSettings } from '../composables/useSettings';
import { useDemoRuntime } from '../demo-runtime';
import { watch, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';

const { settings } = injectSettings();
const runtime = useDemoRuntime();
const { t } = useI18n();

// settings.modemMode の変更を runtime.modemMode に同期
watch(() => settings.value.modemMode, (newMode) => {
  runtime.modemMode.value = newMode;
});

// settings.randomizeSeq の変更を runtime.randomizeSeq に同期
watch(() => settings.value.randomizeSeq, (newValue) => {
  runtime.randomizeSeq.value = newValue;
});

// 初期同期（保存された settings を runtime に反映）
onMounted(() => {
  runtime.modemMode.value = settings.value.modemMode;
  runtime.randomizeSeq.value = settings.value.randomizeSeq;
});
</script>

<template>
  <section class="panel settings-panel">
    <div class="settings-header">
      <h2>{{ $t('settings.title') }}</h2>
    </div>
    <!-- Language Selection Section -->
    <div class="setting-section">
      <h3 class="section-title">{{ $t('settings.language.label') }}</h3>
      <div class="setting-group">
        <div class="language-selector">
          <select v-model="settings.language" class="form-select">
            <option value="auto">{{ $t('settings.language.auto') }}</option>
            <option value="en">{{ $t('settings.language.en') }}</option>
            <option value="ja">{{ $t('settings.language.ja') }}</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Common Section -->
    <div class="setting-section">
      <h3 class="section-title">{{ $t('settings.modem.label') }}</h3>

      <div class="setting-group">
        <div class="mode-selector">
          <button
            class="mode-btn"
            :class="{ active: settings.modemMode === 'dsss' }"
            :disabled="runtime.isBusy.value"
            @click="settings.modemMode = 'dsss'"
          >
            <div class="mode-content">
              <span class="mode-name">{{ $t('settings.modem.dsss') }}</span>
              <span class="mode-desc">Direct Sequence Spread Spectrum</span>
            </div>
          </button>
          <button
            class="mode-btn"
            :class="{ active: settings.modemMode === 'mary' }"
            :disabled="runtime.isBusy.value"
            @click="settings.modemMode = 'mary'"
          >
            <div class="mode-content">
              <span class="mode-name">{{ $t('settings.modem.mary') }}</span>
              <span class="mode-desc">M-ary Modulation (Quadrature)</span>
            </div>
          </button>
        </div>
      </div>

      <div class="setting-group">
        <h4>{{ $t('settings.debug.label') }}</h4>
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.debugMode">
          <span>{{ $t('settings.debug.label') }}</span>
        </label>
        <p class="setting-hint">
          {{ $t('settings.debug.desc') }}
        </p>
      </div>
    </div>

    <!-- Sender Section -->
    <div class="setting-section">
      <h3 class="section-title">{{ $t('common.sender') }}</h3>

      <div class="setting-group">
        <h4>{{ $t('settings.transmission.randomize') }}</h4>
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.randomizeSeq">
          <span>{{ $t('settings.transmission.randomize') }}</span>
        </label>
        <p class="setting-hint">
          {{ $t('settings.transmission.desc') }}
        </p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.settings-panel {
  min-height: auto;
  padding: 0;
}

.settings-header {
  padding: 2rem 2rem 1.5rem;
  background: #fff;
  position: sticky;
  top: 0;
  z-index: 5;
  border-bottom: 1px solid var(--line);
}

.settings-header h2 {
  margin: 0 0 0.25rem 0;
}

.setting-section {
  margin-top: 2.5rem;
  padding: 0 2rem;
}

.setting-section:first-child {
  margin-top: 0;
}

.section-title {
  margin: 0 0 1.25rem;
  font-size: 1.4rem;
  color: var(--ink);
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--primary);
}

.section-title:first-child {
  margin-top: 0;
}

.setting-group {
  margin-top: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--line);
}

.setting-group:last-child {
  border-bottom: none;
}

.setting-group h4 {
  margin: 0 0 0.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: var(--ink);
}

.setting-description {
  margin: 0 0 1rem;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.5;
}

.form-select {
  width: 100%;
  padding: 0.75rem;
  padding-right: 2.5rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fff;
  font-size: 0.95rem;
  color: var(--ink);
  cursor: pointer;
  transition: all 0.2s;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 0.75rem center;
  background-size: 1.25rem;
}

.form-select:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.setting-hint {
  margin: 0.5rem 0 0;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.4;
}

.mode-selector {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.mode-btn {
  border: 2px solid var(--line);
  border-radius: 12px;
  background: #fff;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.mode-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: var(--primary);
}

.mode-btn.active {
  border-color: var(--primary);
  background: #f0f9ff;
  box-shadow: 0 0 0 3px rgba(15, 107, 215, 0.1);
}

.mode-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.mode-content {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.mode-name {
  font-weight: 700;
  font-size: 1rem;
  color: var(--ink);
}

.mode-desc {
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.4;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.95rem;
  color: var(--ink);
}

.checkbox-label input[type="checkbox"] {
  width: 1.1rem;
  height: 1.1rem;
  cursor: pointer;
}

@media (min-width: 780px) {
  .mode-selector {
    flex-direction: row;
  }

  .mode-btn {
    flex: 1;
  }
}
</style>
