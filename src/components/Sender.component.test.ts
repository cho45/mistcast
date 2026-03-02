import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import Sender from './Sender.vue';
import { mountWithRuntime } from '../test/test-helpers';

describe('Sender.vue', () => {
  const TestWrapper = mountWithRuntime(Sender);

  it('should mount without errors', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.exists()).toBe(true);
    expect(wrapper.find('.sender-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('Sender');
  });

  it('should have initial state', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.find('textarea').exists()).toBe(true);
    expect(wrapper.find('textarea').element.value).toBe('Hello Acoustic World!');
  });

  it('should toggle drag state on drag events', async () => {
    const wrapper = mount(TestWrapper);

    const panel = wrapper.find('.sender-panel');

    await panel.trigger('dragenter', { preventDefault: () => {} });
    expect(panel.classes()).toContain('is-dragging');

    await panel.trigger('dragleave', { preventDefault: () => {} });
    expect(panel.classes()).not.toContain('is-dragging');

    await panel.trigger('dragenter', { preventDefault: () => {} });
    expect(panel.classes()).toContain('is-dragging');

    const mockFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    await panel.trigger('drop', {
      preventDefault: () => {},
      dataTransfer: { files: [mockFile] },
    });
    expect(panel.classes()).not.toContain('is-dragging');
  });
});
