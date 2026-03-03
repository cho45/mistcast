<script setup lang="ts">
import { injectSettings } from '../composables/useSettings';
import { useDemoRuntime } from '../demo-runtime';
import { watch, onMounted } from 'vue';

const { settings } = injectSettings();
const runtime = useDemoRuntime();

// settings.modemMode の変更を runtime.modemMode に同期
watch(() => settings.value.modemMode, (newMode) => {
  runtime.modemMode.value = newMode;
});

// settings.randomizeSeq の変更を runtime.randomizeSeq に同期
watch(() => settings.value.randomizeSeq, (newValue) => {
  runtime.randomizeSeq.value = newValue;
});

// 初期同期（runtime の値を settings に反映）
onMounted(() => {
  settings.value.modemMode = runtime.modemMode.value;
  settings.value.randomizeSeq = runtime.randomizeSeq.value;
});
</script>

<template>
  <section class="panel settings-panel">
    <div class="settings-header">
      <h2>Settings</h2>
    </div>

    <!-- Common Section -->
    <div class="setting-section">
      <h3 class="section-title">Common</h3>

      <div class="setting-group">
        <h4>Modem Mode</h4>
        <p class="setting-description">
          Select the modulation scheme for acoustic data transmission.
        </p>

        <div class="mode-selector">
          <button
            class="mode-btn"
            :class="{ active: settings.modemMode === 'dsss' }"
            :disabled="runtime.isBusy.value"
            @click="settings.modemMode = 'dsss'"
          >
            <div class="mode-content">
              <span class="mode-name">DSSS (Slow)</span>
              <span class="mode-desc">Direct Sequence Spread Spectrum - Robust but slower</span>
            </div>
          </button>
          <button
            class="mode-btn"
            :class="{ active: settings.modemMode === 'mary' }"
            :disabled="runtime.isBusy.value"
            @click="settings.modemMode = 'mary'"
          >
            <div class="mode-content">
              <span class="mode-name">M-ARY (Fast)</span>
              <span class="mode-desc">M-ary Modulation - Faster transmission</span>
            </div>
          </button>
        </div>
      </div>

      <div class="setting-group">
        <h4>Debug</h4>
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.debugMode">
          <span>Debug Mode</span>
        </label>
        <p class="setting-hint">
          Enable verbose logging and debug information
        </p>
      </div>
    </div>

    <!-- Sender Section -->
    <div class="setting-section">
      <h3 class="section-title">Sender</h3>

      <div class="setting-group">
        <h4>Transmission Options</h4>
        <label class="checkbox-label">
          <input type="checkbox" v-model="settings.randomizeSeq">
          <span>Randomized Sequence</span>
        </label>
        <p class="setting-hint">
          より堅牢な伝送のためにランダム化する (重複パケット回避)
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
