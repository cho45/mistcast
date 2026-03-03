import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import Receiver from './Receiver.vue';
import { mountWithRuntime } from '../test/test-helpers';

describe('Receiver.vue', () => {
  const TestWrapper = mountWithRuntime(Receiver);

  it('should mount without errors', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.exists()).toBe(true);
    expect(wrapper.find('.receiver-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('Receiver');
  });

  it('should have initial state', () => {
    const wrapper = mount(TestWrapper);

    // 初期状態ではmetric-gridが表示される
    expect(wrapper.find('.metric-grid').exists()).toBe(true);

    // .display要素はprogressPercent >= 1.0の場合のみ表示される
    // 初期状態では表示されない
    expect(wrapper.find('.display').exists()).toBe(false);
  });

  it('should render mic toggle button', () => {
    const wrapper = mount(TestWrapper);

    const micButton = wrapper.findAll('button').find(btn => btn.text().includes('Enable Mic'));
    expect(micButton).toBeDefined();
  });

  describe('Debug Mode', () => {
    it('should not show proc-stats when debugMode is false', () => {
      const wrapper = mount(TestWrapper);

      // debugModeがfalseの場合、proc-statsは表示されない
      expect(wrapper.find('.proc-stats').exists()).toBe(false);
    });

    it('should not show rx-log when debugMode is false', () => {
      const wrapper = mount(TestWrapper);

      // debugModeがfalseの場合、rx-logは表示されない
      expect(wrapper.find('.rx-log').exists()).toBe(false);
    });

    it('should show proc-stats when debugMode is true', async () => {
      const wrapper = mount(TestWrapper);
      const receiver = wrapper.findComponent(Receiver);

      // debugModeをtrueに設定
      receiver.vm.settings.debugMode = true;
      await receiver.vm.$nextTick();

      // proc-statsが表示されることを確認
      expect(wrapper.find('.proc-stats').exists()).toBe(true);
    });

    it('should show rx-log when debugMode is true and logs exist', async () => {
      const wrapper = mount(TestWrapper);
      const receiver = wrapper.findComponent(Receiver);

      // debugModeをtrueに設定
      receiver.vm.settings.debugMode = true;

      // rxLogsを追加
      receiver.vm.rxLogs = ['test log'];
      await receiver.vm.$nextTick();

      // rx-logが表示されることを確認
      expect(wrapper.find('.rx-log').exists()).toBe(true);
    });
  });
});
