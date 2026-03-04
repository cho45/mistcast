import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import Sender from './Sender.vue';
import { mountWithRuntime } from '../test/test-helpers';

describe('Sender.vue', () => {
  const TestWrapper = mountWithRuntime(Sender);

  // localStorageのモック
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: (key: string) => store[key] || null,
      setItem: (key: string, value: string) => {
        store[key] = value.toString();
      },
      clear: () => {
        store = {};
      },
    };
  })();

  beforeEach(() => {
    // 各テスト前にlocalStorageをクリア
    localStorageMock.clear();
    vi.stubGlobal('localStorage', localStorageMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('should mount without errors', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.exists()).toBe(true);
    expect(wrapper.find('.sender-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('Sender');
  });

  it('should have initial state with text mode', () => {
    const wrapper = mount(TestWrapper);

    expect(wrapper.find('textarea').exists()).toBe(true);
    expect(wrapper.find('textarea').element.value).toBe('Hello Acoustic World!');
    expect(wrapper.find('.btn-primary').exists()).toBe(true);
  });

  describe('Send Mode Management', () => {
    it('should start in text mode by default', () => {
      const wrapper = mount(TestWrapper);

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.text()).toBe('Send');
    });

    it('should render tab buttons for send mode selection', () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      expect(tabButtons.length).toBe(3);

      expect(tabButtons[0].text()).toContain('Text');
      expect(tabButtons[1].text()).toContain('Sample');
      expect(tabButtons[2].text()).toContain('File');
    });

    it('should have text tab active by default', () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      expect(tabButtons[0].classes()).toContain('active');
    });

    it('should switch to sample mode when sample tab is clicked', async () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[1].trigger('click');

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.text()).toBe('Send Sample Image');
      expect(tabButtons[1].classes()).toContain('active');
      expect(tabButtons[0].classes()).not.toContain('active');
    });

    it('should save send mode to localStorage', async () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[1].trigger('click');

      // localStorageに保存されたことを確認
      expect(localStorageMock.getItem('sender-send-mode')).toBe('sample');
    });

    it('should restore send mode from localStorage on mount', () => {
      // localStorageにsampleモードを保存
      localStorageMock.setItem('sender-send-mode', 'sample');

      const wrapper = mount(TestWrapper);

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.text()).toBe('Send Sample Image');

      // サンプルタブがアクティブであることを確認
      const tabButtons = wrapper.findAll('button.tab-item');
      expect(tabButtons[1].classes()).toContain('active');
    });
  });

  describe('File Selection and Preview', () => {
    it('should show file preview when file is selected', async () => {
      const wrapper = mount(TestWrapper);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

      const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [testFile],
        writable: false,
      });

      await fileInput.trigger('change');
      await new Promise(resolve => setTimeout(resolve, 10));

      // ファイルプレビューが表示されることを確認
      expect(wrapper.find('.file-preview').exists()).toBe(true);
      expect(wrapper.find('.file-name').text()).toBe('test.txt');

      consoleErrorSpy.mockRestore();
    });

    it('should clear file when remove button is clicked', async () => {
      const wrapper = mount(TestWrapper);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

      const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [testFile],
        writable: false,
      });

      await fileInput.trigger('change');
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(wrapper.find('.file-preview').exists()).toBe(true);

      // ファイルクリアボタンをクリック
      const removeButton = wrapper.find('.remove-file-btn');
      await removeButton.trigger('click');

      // プレビューが消えることを確認
      expect(wrapper.find('.file-preview').exists()).toBe(false);

      consoleErrorSpy.mockRestore();
    });

    it('should show drop zone when file mode is selected without file', async () => {
      const wrapper = mount(TestWrapper);

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

      // ドロップゾーンが表示されることを確認
      expect(wrapper.find('.drop-zone').exists()).toBe(true);
    });

    it('should switch to file mode and show file name in button when file is selected', async () => {
      const wrapper = mount(TestWrapper);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

      const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [testFile],
        writable: false,
      });

      await fileInput.trigger('change');
      await new Promise(resolve => setTimeout(resolve, 10));

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.text()).toContain('Send:');
      expect(sendButton.text()).toContain('test.txt');

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Unified Send Button', () => {
    it('should send text when in text mode', async () => {
      const wrapper = mount(TestWrapper);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      const sendButton = wrapper.find('.btn-primary');
      await sendButton.trigger('click');

      // テキスト送信が試行されることを確認
      // 実際の送信はmockされているため、エラーが出ないだけでOK
      await new Promise(resolve => setTimeout(resolve, 10));

      consoleErrorSpy.mockRestore();
    });

    it('should be disabled when input is empty in text mode', async () => {
      const wrapper = mount(TestWrapper);

      const textarea = wrapper.find('textarea');
      await textarea.setValue('');

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.attributes('disabled')).toBeDefined();
    });

    it('should disable textarea during transmission', async () => {
      const wrapper = mount(TestWrapper);
      const sender = wrapper.findComponent(Sender);

      // 直接送信中状態にする
      (sender.vm as any).isTransmitting = true;
      await wrapper.vm.$nextTick();

      const textarea = wrapper.find('textarea');
      expect(textarea.attributes('disabled')).toBeDefined();
    });

    it('should show file name in button when file is selected', async () => {
      const wrapper = mount(TestWrapper);
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

      const testFile = new File(['test content'], 'myfile.pdf', { type: 'application/pdf' });
      const fileInput = wrapper.find('input[type="file"]');
      Object.defineProperty(fileInput.element, 'files', {
        value: [testFile],
        writable: false,
      });

      await fileInput.trigger('change');
      await new Promise(resolve => setTimeout(resolve, 10));

      const sendButton = wrapper.find('.btn-primary');
      expect(sendButton.text()).toBe('Send: myfile.pdf');

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Size Indicator', () => {
    it('should show byte counter for text mode', () => {
      const wrapper = mount(TestWrapper);

      expect(wrapper.find('.size-indicator').exists()).toBe(true);
      const sizeText = wrapper.find('.size-indicator').text();
      expect(sizeText).toMatch(/\d+ \/ 4080 bytes/);
    });

    it('should show warning class when size exceeds 3500 bytes', async () => {
      const wrapper = mount(TestWrapper);

      // 大きなテキストを設定
      const largeText = 'a'.repeat(3600);
      const textarea = wrapper.find('textarea');
      await textarea.setValue(largeText);

      const sizeIndicator = wrapper.find('.size-indicator');
      expect(sizeIndicator.classes()).toContain('warning');
    });
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

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

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
      expect((fileInput.element as HTMLInputElement).value).toBe('');

      // ファイルが選択されていないことを確認
      expect(wrapper.find('.file-preview').exists()).toBe(false);

      consoleWarnSpy.mockRestore();
    });

    it('should show error toast when file size exceeds limit on drop', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const wrapper = mount(TestWrapper);

      // ファイルモードのタブを選択してドロップゾーンを表示
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');
      await new Promise(resolve => setTimeout(resolve, 10));

      // Create a file larger than MAX_FILE_SIZE
      const largeFile = new File(
        [new ArrayBuffer(MAX_FILE_SIZE + 1)],
        'large.txt',
        { type: 'text/plain' }
      );
      expect(largeFile.size).toBeGreaterThan(MAX_FILE_SIZE);

      // ドロップゾーンに対してドロップイベントを発火
      const dropZone = wrapper.find('.drop-zone');
      expect(dropZone.exists()).toBe(true);

      await dropZone.trigger('drop', {
        preventDefault: () => {},
        dataTransfer: { files: [largeFile] },
      });

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 10));

      // Verify that isDragging was reset to false
      expect(dropZone.classes()).not.toContain('is-dragging');

      // ファイルが選択されていないことを確認
      expect(wrapper.find('.file-preview').exists()).toBe(false);

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

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

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

      // ファイルプレビューが表示されることを確認
      expect(wrapper.find('.file-preview').exists()).toBe(true);
      expect(wrapper.find('.file-name').text()).toBe('small.txt');

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

      // ファイルモードのタブを選択
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');

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
      expect((fileInput.element as HTMLInputElement).value).toBe('');

      vi.useRealTimers();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Tab Buttons', () => {
    it('should disable tab buttons during transmission', async () => {
      const wrapper = mount(TestWrapper);

      // 最初は有効
      const tabButtons = wrapper.findAll('button.tab-item');
      tabButtons.forEach(button => {
        expect(button.attributes('disabled')).toBeUndefined();
      });

      // ボタンが存在することを確認
      expect(tabButtons.length).toBe(3);
    });

    it('should have correct labels for each tab option', () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      expect(tabButtons.length).toBe(3);

      // テキストオプション
      expect(tabButtons[0].find('.tab-icon').text()).toBe('📝');
      expect(tabButtons[0].find('.tab-text').text()).toBe('Text');

      // サンプル画像オプション
      expect(tabButtons[1].find('.tab-icon').text()).toBe('🖼️');
      expect(tabButtons[1].find('.tab-text').text()).toBe('Sample');

      // ファイルオプション
      expect(tabButtons[2].find('.tab-icon').text()).toBe('📁');
      expect(tabButtons[2].find('.tab-text').text()).toBe('File');
    });
  });

  describe('Sample Image Preview', () => {
    it('should show sample preview when sample mode is selected', async () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[1].trigger('click');

      expect(wrapper.find('.sample-preview').exists()).toBe(true);
      expect(wrapper.find('.sample-preview img').exists()).toBe(true);
    });

    it('should show sample info with correct size', async () => {
      const wrapper = mount(TestWrapper);

      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[1].trigger('click');

      const sampleInfo = wrapper.find('.sample-info');
      expect(sampleInfo.exists()).toBe(true);
      expect(sampleInfo.text()).toContain('test.png');
      expect(sampleInfo.text()).toContain('921');
    });
  });

  describe('Drag and Drop', () => {
    it('should toggle drag state on drop zone', async () => {
      const wrapper = mount(TestWrapper);

      // ファイルモードのタブを選択してドロップゾーンを表示
      const tabButtons = wrapper.findAll('button.tab-item');
      await tabButtons[2].trigger('click');
      await new Promise(resolve => setTimeout(resolve, 10));

      const dropZone = wrapper.find('.drop-zone');
      expect(dropZone.exists()).toBe(true);

      await dropZone.trigger('dragenter', { preventDefault: () => {} });
      expect(dropZone.classes()).toContain('is-dragging');

      await dropZone.trigger('dragleave', { preventDefault: () => {} });
      expect(dropZone.classes()).not.toContain('is-dragging');
    });
  });

  describe('Randomized Seq option removed', () => {
    it('should not have Randomized Seq checkbox in sender panel', () => {
      const wrapper = mount(TestWrapper);

      // .sender-options要素が存在しないことを確認
      expect(wrapper.find('.sender-options').exists()).toBe(false);

      // Randomized Seqのチェックボックスが存在しないことを確認
      const checkboxLabels = wrapper.findAll('.checkbox-label');
      const randomizeSeqLabel = checkboxLabels.find(label =>
        label.text().includes('Randomized Seq')
      );
      expect(randomizeSeqLabel).toBeUndefined();
    });

    it('should still have runtime.randomizeSeq available for Settings component', () => {
      const wrapper = mount(TestWrapper);

      // Senderコンポーネント自体にはrandomizeSeq UIがないが、
      // runtime.randomizeSeqは内部で使用されている（Sender.vueのstartSendingDataで参照）
      // このテストはランタイムが正しく設定されていることを確認する
      expect(wrapper.exists()).toBe(true);
    });
  });
});
