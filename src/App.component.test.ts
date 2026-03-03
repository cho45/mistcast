import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import App from './App.vue';

describe('App.vue - App Header and Dialog', () => {
  it('should render app-header instead of hero', () => {
    const wrapper = mount(App);

    const appHeader = wrapper.find('.app-header');
    expect(appHeader.exists()).toBe(true);
  });

  it('should not render hero section', () => {
    const wrapper = mount(App);

    const hero = wrapper.find('.hero');
    expect(hero.exists()).toBe(false);
  });

  it('should render settings trigger button with gear icon', () => {
    const wrapper = mount(App);

    const settingsTrigger = wrapper.find('.settings-trigger');
    expect(settingsTrigger.exists()).toBe(true);
    expect(settingsTrigger.attributes('aria-label')).toBe('Open Settings');
  });

  it('should render settings dialog', () => {
    const wrapper = mount(App);

    const settingsDialog = wrapper.find('.settings-dialog');
    expect(settingsDialog.exists()).toBe(true);
  });

  it('should have title as Mistcast', () => {
    const wrapper = mount(App);

    const appHeader = wrapper.find('.app-header');
    const h1 = appHeader.find('h1');
    expect(h1.exists()).toBe(true);
    expect(h1.text()).toBe('Mistcast');
  });
});

describe('App.vue - 2 Tab Structure', () => {
  it('should have activeTab type as sender or receiver only', () => {
    const wrapper = mount(App);

    // activeTabが'sender' | 'receiver'型であることを確認
    expect(['sender', 'receiver']).toContain(wrapper.vm.activeTab);
  });

  it('should render app correctly', () => {
    const wrapper = mount(App);

    // App shellが存在することを確認
    expect(wrapper.find('.app-shell').exists()).toBe(true);
  });

  it('should not have Settings tab button in template', () => {
    const wrapper = mount(App);

    // Settingsタブボタンのクリックハンドラ "activeTab = 'settings'" が存在しないことを確認
    const html = wrapper.html();
    expect(html).not.toContain("activeTab = 'settings'");
  });

  it('should render Sender and Receiver tabs after initialization', async () => {
    const wrapper = mount(App);

    // 初期状態ではタブが表示されない
    expect(wrapper.findAll('.tab-btn')).toHaveLength(0);

    // 初期化ボタンをクリック
    const initButton = wrapper.find('.btn-primary');
    expect(initButton.exists()).toBe(true);
    await initButton.trigger('click');

    // 非同期処理を待つ
    await new Promise(resolve => setTimeout(resolve, 50));
    await wrapper.vm.$nextTick();

    // coreReadyがtrueになっていることを確認
    expect(wrapper.vm.runtime.coreReady.value).toBe(true);

    // タブが表示されることを確認
    const tabBtns = wrapper.findAll('.tab-btn');
    expect(tabBtns).toHaveLength(2);
    expect(tabBtns[0].text()).toBe('Sender');
    expect(tabBtns[1].text()).toBe('Receiver');
  });

  it('should switch between sender and receiver tabs', async () => {
    const wrapper = mount(App);

    // 初期化
    const initButton = wrapper.find('.btn-primary');
    await initButton.trigger('click');
    await new Promise(resolve => setTimeout(resolve, 50));
    await wrapper.vm.$nextTick();

    const tabBtns = wrapper.findAll('.tab-btn');

    // 初期状態はreceiver
    expect(wrapper.vm.activeTab).toBe('receiver');

    // Senderタブをクリック
    await tabBtns[0].trigger('click');
    expect(wrapper.vm.activeTab).toBe('sender');

    // Receiverタブをクリック
    await tabBtns[1].trigger('click');
    expect(wrapper.vm.activeTab).toBe('receiver');
  });
});

describe('App.vue - Dialog Interaction', () => {
  it('should open settings dialog when trigger is clicked', async () => {
    const wrapper = mount(App);
    const showModalSpy = vi.fn();

    wrapper.vm.settingsDialog = {
      showModal: showModalSpy,
      close: vi.fn(),
      getBoundingClientRect: vi.fn(() => ({ left: 0, right: 100, top: 0, bottom: 100 }))
    } as unknown as HTMLDialogElement;

    const settingsTrigger = wrapper.find('.settings-trigger');
    await settingsTrigger.trigger('click');

    expect(showModalSpy).toHaveBeenCalled();
  });

  it('should close settings dialog when close button is clicked', async () => {
    const wrapper = mount(App);
    const closeSpy = vi.fn();

    wrapper.vm.settingsDialog = {
      showModal: vi.fn(),
      close: closeSpy,
      getBoundingClientRect: vi.fn(() => ({ left: 0, right: 100, top: 0, bottom: 100 }))
    } as unknown as HTMLDialogElement;

    const closeButton = wrapper.find('.dialog-close');
    await closeButton.trigger('click');

    expect(closeSpy).toHaveBeenCalled();
  });
});
