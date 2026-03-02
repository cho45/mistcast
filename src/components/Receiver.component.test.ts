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

    expect(wrapper.find('.display').exists()).toBe(true);
    expect(wrapper.find('.placeholder').exists()).toBe(true);
    expect(wrapper.find('.placeholder').text()).toBe('Waiting for synchronization...');
  });

  it('should render mic toggle button', () => {
    const wrapper = mount(TestWrapper);

    const micButton = wrapper.findAll('button').find(btn => btn.text().includes('Enable Mic'));
    expect(micButton).toBeDefined();
  });
});
