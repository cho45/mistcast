import { describe, it, expect, vi } from 'vitest';
import Receiver from './Receiver.vue';
import { mountApp } from '../test/test-helpers';

describe('Receiver.vue', () => {
  it('should display correct localized status', async () => {
    const wrapper = mountApp(Receiver);
    const vm = wrapper.vm as any;

    expect(wrapper.find('.status-chip').text()).toBe('Idle');

    vm.receiverStatus = 'mic-active-rx';
    await wrapper.vm.$nextTick();
    expect(wrapper.find('.status-chip').text()).toBe('Ready');

    vm.receiverStatus = 'receiving';
    await wrapper.vm.$nextTick();
    expect(wrapper.find('.status-chip').text()).toBe('Receiving...');
  });

  it('should mount without errors', () => {
    const wrapper = mountApp(Receiver);

    expect(wrapper.exists()).toBe(true);
    expect(wrapper.find('.receiver-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('Receiver');
  });

  it('should have initial state', () => {
    const wrapper = mountApp(Receiver);

    // デフォルト（Debug Mode: false）では metric-grid は表示されない
    expect(wrapper.find('.metric-grid').exists()).toBe(false);

    // .display要素は progressPercent >= 1.0 の場合のみ表示される
    // 初期状態では表示されない
    expect(wrapper.find('.display').exists()).toBe(false);
  });

  it('should render input mode tabs and clear button', () => {
    const wrapper = mountApp(Receiver);

    // 入力モードのタブボタンを確認
    const loopbackTab = wrapper.findAll('.mode-tabs button').find(btn => btn.text().includes('Loopback'));
    const micTab = wrapper.findAll('.mode-tabs button').find(btn => btn.text().includes('Microphone'));
    expect(loopbackTab).toBeDefined();
    expect(micTab).toBeDefined();

    // クリアボタンを確認
    const clearButton = wrapper.find('.btn-clear-action');
    expect(clearButton.exists()).toBe(true);
    expect(clearButton.text()).includes('Clear & Reset');
  });

  describe('Debug Mode', () => {
    it('should not show proc-stats when debugMode is false', () => {
      const wrapper = mountApp(Receiver);

      // debugModeがfalseの場合、proc-statsは表示されない
      expect(wrapper.find('.proc-stats').exists()).toBe(false);
    });

    it('should not show rx-log when debugMode is false', () => {
      const wrapper = mountApp(Receiver);

      // debugModeがfalseの場合、rx-logは表示されない
      expect(wrapper.find('.rx-log').exists()).toBe(false);
    });

    it('should show metric-grid and proc-stats when debugMode is true', async () => {
      const wrapper = mountApp(Receiver);
      const receiver = wrapper.findComponent(Receiver);

      // debugModeをtrueに設定
      const vm = receiver.vm as { settings: { debugMode: boolean } };
      vm.settings.debugMode = true;
      await receiver.vm.$nextTick();

      // metric-grid と proc-stats が表示されることを確認
      expect(wrapper.find('.metric-grid').exists()).toBe(true);
      expect(wrapper.find('.proc-stats').exists()).toBe(true);
    });

    it('should show rx-log when debugMode is true and logs exist', async () => {
      const wrapper = mountApp(Receiver);
      const receiver = wrapper.findComponent(Receiver);

      // debugModeをtrueに設定
      const vm = receiver.vm as { settings: { debugMode: boolean }; rxLogs: string[] };
      vm.settings.debugMode = true;

      // rxLogsを追加
      vm.rxLogs = ['test log'];
      await receiver.vm.$nextTick();

      // rx-logが表示されることを確認
      expect(wrapper.find('.rx-log').exists()).toBe(true);
    });
  });
});
