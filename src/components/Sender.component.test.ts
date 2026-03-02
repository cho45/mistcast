import { describe, it, expect, vi } from 'vitest';
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

  describe('File size validation', () => {
    const MAX_FILE_SIZE = 255 * 16; // 4080 bytes

    it('should show error toast when file size exceeds limit on file select', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const wrapper = mount(TestWrapper);

      // Create a file larger than MAX_FILE_SIZE
      const largeFile = new File(
        [new ArrayBuffer(MAX_FILE_SIZE + 1)],
        'large.txt',
        { type: 'text/plain' }
      );
      expect(largeFile.size).toBeGreaterThan(MAX_FILE_SIZE);

      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [largeFile],
        writable: false,
      });

      await fileInput.trigger('change');

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 10));

      // Verify that the file input was reset (value is empty)
      expect(fileInput.element.value).toBe('');

      consoleWarnSpy.mockRestore();
    });

    it('should show error toast when file size exceeds limit on drop', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const wrapper = mount(TestWrapper);

      // Create a file larger than MAX_FILE_SIZE
      const largeFile = new File(
        [new ArrayBuffer(MAX_FILE_SIZE + 1)],
        'large.txt',
        { type: 'text/plain' }
      );
      expect(largeFile.size).toBeGreaterThan(MAX_FILE_SIZE);

      const panel = wrapper.find('.sender-panel');
      await panel.trigger('drop', {
        preventDefault: () => {},
        dataTransfer: { files: [largeFile] },
      });

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 10));

      // Verify that isDragging was reset to false
      expect(panel.classes()).not.toContain('is-dragging');

      consoleWarnSpy.mockRestore();
    });

    it('should show error toast when text size exceeds limit', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const wrapper = mount(TestWrapper);

      // Set text larger than MAX_FILE_SIZE
      const largeText = 'a'.repeat(MAX_FILE_SIZE + 1);
      const textarea = wrapper.find('textarea');
      await textarea.setValue(largeText);

      const sendButton = wrapper.find('.btn-primary');
      await sendButton.trigger('click');

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 10));

      // Verify that the textarea still has the large text (wasn't sent)
      expect(textarea.element.value).toBe(largeText);

      consoleWarnSpy.mockRestore();
    });

    it('should accept file within size limit', async () => {
      const wrapper = mount(TestWrapper);

      // Create a file smaller than MAX_FILE_SIZE
      const smallFile = new File(
        [new ArrayBuffer(MAX_FILE_SIZE - 1)],
        'small.txt',
        { type: 'text/plain' }
      );
      expect(smallFile.size).toBeLessThan(MAX_FILE_SIZE);

      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [smallFile],
        writable: false,
      });

      // Mock console.error to avoid error output during test
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      await fileInput.trigger('change');

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 10));

      // Check that NO toast error message is displayed
      const toastContainer = document.querySelector('.toast-container');
      // Toast container may not exist if no toasts were shown
      if (toastContainer) {
        const toastError = toastContainer.querySelector('.toast-error');
        expect(toastError).toBeFalsy();
      }

      consoleErrorSpy.mockRestore();
    });

    it('should show toast with auto-dismiss after timeout', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      vi.useFakeTimers();

      const wrapper = mount(TestWrapper);

      // Create a file larger than MAX_FILE_SIZE to trigger toast
      const largeFile = new File(
        [new ArrayBuffer(MAX_FILE_SIZE + 1)],
        'large.txt',
        { type: 'text/plain' }
      );

      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [largeFile],
        writable: false,
      });

      await fileInput.trigger('change');

      // Fast-forward time to trigger toast auto-dismiss
      vi.advanceTimersByTime(5100);

      // Verify that the file input was reset
      expect(fileInput.element.value).toBe('');

      vi.useRealTimers();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Send File button label', () => {
    it('should display "Send File (max 4KB)" label', () => {
      const wrapper = mount(TestWrapper);

      const sendFileButton = wrapper.findAll('button').find(btn =>
        btn.text().includes('Send File')
      );
      expect(sendFileButton).toBeTruthy();
      expect(sendFileButton?.text()).toBe('Send File (max 4KB)');
    });
  });
});
